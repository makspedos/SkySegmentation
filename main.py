import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import segmentation_refinement as refine
from tqdm import tqdm
from pathlib import Path

MOTION_SCALE = 0.5  # Global scale that slows virtual-camera motion

class Mask2FormerSeg(torch.nn.Module):
    """Semantic segmenter that outputs a binary sky mask for each input image."""

    def __init__(
        self,
        model: str = "facebook/mask2former-swin-large-cityscapes-semantic",
        device: str = "cuda",
    ):
        super().__init__()
        self.proc = AutoImageProcessor.from_pretrained(model)
        self.net = Mask2FormerForUniversalSegmentation.from_pretrained(model)
        self.net.to(device).eval()
        id2label = self.net.config.id2label
        self.sky_id = next(i for i, l in id2label.items() if "sky" in l.lower())
        self.device = device

    @torch.inference_mode()
    def forward(self, imgs):
        """Return 0/255 sky-only masks for a batch of RGB images."""
        inputs = self.proc(imgs, return_tensors="pt").to(self.device)
        outputs = self.net(**inputs)
        sems = self.proc.post_process_semantic_segmentation(
            outputs, target_sizes=[im.shape[:2] for im in imgs]
        )
        return [
            (s == self.sky_id).cpu().numpy().astype(np.uint8) * 255 for s in sems
        ]


class RefineMask:
    """Cleans up coarse sky masks with morphology and optional cascade refinement."""

    def __init__(self, mask=None, cascade: bool = False):
        self.mask = mask
        self.cascade = refine.Refiner(device="cuda:0") if cascade else None

    def _morph(self, mask):
        """Morphologically smooth the mask (internal helper)."""
        h = mask.shape[0]
        k = max(3, h // 100 | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        erode = cv2.erode(mask, kernel, iterations=2)
        return cv2.dilate(erode, kernel, iterations=1)

    @staticmethod
    def _component_filter(mask):
        """Keep the largest connected sky components, discard tiny blobs."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
        cleaned = np.zeros_like(mask)
        for c in cnts:
            cv2.drawContours(cleaned, [c], -1, 255, thickness=cv2.FILLED)
        return cleaned

    def refine(self, image=None):
        """Return an improved sky mask for the associated video frame."""
        if self.mask is None:
            raise RuntimeError("mask not set")
        mask = (
            cv2.imread(self.mask, cv2.IMREAD_GRAYSCALE)
            if isinstance(self.mask, (str, Path))
            else self.mask.copy()
        )
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if image is not None and self.cascade is not None:
            return self.cascade.refine(image, mask, fast=True, L=900)
        return self._component_filter(self._morph(mask))


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Return logical NOT of a 0/255 binary mask."""
    return cv2.bitwise_not(mask.astype(np.uint8))


def rot_matrix(axis, angle):
    """Return a 3×3 rotation matrix for a given axis ('x','y','z') and angle in radians."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError("bad axis")


def sky_crop(
    panorama,
    yaw_deg,
    pitch_deg,
    roll_deg,
    fov_x_deg=90,
    fov_y_deg=60,
    crop_w=512,
    crop_h=512,
):
    """Extract a perspective view from an equirectangular sky panorama."""
    fov_x = np.deg2rad(fov_x_deg)
    fov_y = np.deg2rad(fov_y_deg)
    yaw, pitch, roll = map(np.deg2rad, (yaw_deg, pitch_deg, roll_deg))
    x = np.linspace(-np.tan(fov_x / 2), np.tan(fov_x / 2), crop_w)
    y = np.linspace(np.tan(fov_y / 2), -np.tan(fov_y / 2), crop_h)
    xv, yv = np.meshgrid(x, y)
    zv = -np.ones_like(xv)
    dirs = np.stack([xv, yv, zv], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    R = rot_matrix("y", yaw) @ rot_matrix("x", pitch) @ rot_matrix("z", roll)
    dirs_rot = dirs @ R.T
    theta = np.arctan2(dirs_rot[..., 0], -dirs_rot[..., 2])
    phi = np.arcsin(dirs_rot[..., 1])
    u = (theta + np.pi) / (2 * np.pi)
    v = 1 - (phi + np.pi / 2) / np.pi
    h, w = panorama.shape[:2]
    map_x = (u * w).astype(np.float32)
    map_y = (v * h).astype(np.float32)
    return cv2.remap(
        panorama, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
    )


def update_camera_angles(yaw, pitch, dx, dy, frame_w, frame_h, fov_x, fov_y):
    """Integrate optical-flow translation into new yaw/pitch angles (degrees)."""
    yaw = (yaw + (dx / frame_w) * fov_x * MOTION_SCALE) % 360
    pitch = (pitch + (dy / frame_h) * fov_y * MOTION_SCALE) % 360
    return yaw, pitch


def color_transfer(img, target):
    """Match global LAB statistics of img to those of target."""
    lab_src = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_tar = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    m_s, s_s = cv2.meanStdDev(lab_src)
    m_t, s_t = cv2.meanStdDev(lab_tar)
    res = (lab_src - m_s.reshape(1, 1, 3)) / (s_s.reshape(1, 1, 3) + 1e-8)
    res = res * s_t.reshape(1, 1, 3) + m_t.reshape(1, 1, 3)
    res = np.clip(res, 0, 255).astype(np.uint8)
    return cv2.cvtColor(res, cv2.COLOR_LAB2BGR)


def blend_by_mask(bg, fg, mask):
    """Alpha-blend two images using a single-channel 0/255 mask."""
    a = (mask.astype(np.float32) / 255.0)[..., None]
    return (bg.astype(np.float32) * (1 - a) + fg.astype(np.float32) * a).astype(
        np.uint8
    )


def prepare_gray_laplacian(bgr):
    """Return an edge-rich single-channel version of an image for optical flow."""
    return cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_8U, ksize=7)


def replace_sky_sequence(
    frames,
    sky_panorama,
    refined_masks,
    init_orientation=(90.0, 90.0, 0.0),
    fov=50,
):
    """Produce a list of frames where the original sky is replaced by a moving panorama."""
    fov_x = fov_y = fov
    yaw, pitch, roll = init_orientation
    shi_params = dict(maxCorners=50, qualityLevel=0.65, minDistance=50, blockSize=7)
    lk_params = dict(
        winSize=(30, 30),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    first = frames[0]
    prev_gray = prepare_gray_laplacian(first)
    prev_corners = cv2.goodFeaturesToTrack(
        prev_gray, mask=invert_mask(refined_masks[0]), **shi_params
    )
    out_frames = []
    for i in tqdm(range(len(frames)), desc="Replacing sky"):
        frame = frames[i]
        cur_gray = prepare_gray_laplacian(frame)
        next_corners, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, prev_corners, None, **lk_params
        )
        if next_corners is None or st.sum() < 3:
            out_frames.append(frame.copy())
            prev_gray = cur_gray
            prev_corners = cv2.goodFeaturesToTrack(
                prev_gray, mask=invert_mask(refined_masks[i]), **shi_params
            )
            continue
        matched_next = next_corners[st == 1]
        matched_prev = prev_corners[st == 1]
        dy = np.median(matched_next[:, 1] - matched_prev[:, 1])
        dx = np.median(matched_next[:, 0] - matched_prev[:, 0])
        yaw, pitch = update_camera_angles(
            yaw, pitch, dx, dy, frame.shape[1], frame.shape[0], fov_x, fov_y
        )
        M, _ = cv2.estimateAffinePartial2D(
            matched_prev, matched_next, method=cv2.LMEDS
        )
        if M is not None:
            roll -= np.degrees(np.arctan2(M[0, 1], M[0, 0])) * MOTION_SCALE
        sky_view = sky_crop(
            sky_panorama,
            yaw,
            pitch,
            roll,
            fov_x,
            fov_y,
            frame.shape[1],
            frame.shape[0],
        )
        frame_ct = color_transfer(frame, sky_view)
        blended = blend_by_mask(frame_ct, sky_view, refined_masks[i])
        out_frames.append(blended)
        prev_gray = cur_gray
        if len(matched_next) < 6 or i % 5 == 0:
            prev_corners = cv2.goodFeaturesToTrack(
                prev_gray, mask=invert_mask(refined_masks[i]), **shi_params
            )
        else:
            prev_corners = matched_next.reshape(-1, 1, 2)
    return out_frames


def process_video_sky_replacement(
    video_path: str | Path,
    sky_image_path: str | Path,
    output_path: str | Path,
    scale: float = 0.5,
):
    """End-to-end routine: load video, swap its sky, and write the result."""
    video_path, sky_image_path, output_path = map(Path, (video_path, sky_image_path, output_path))
    sky = cv2.imread(str(sky_image_path), cv2.IMREAD_COLOR)
    if sky is None:
        raise FileNotFoundError(f"cannot read {sky_image_path}")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Extracting frames") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            frames.append(frame)
            pbar.update(1)
    cap.release()
    if not frames:
        raise RuntimeError("no frames extracted")
    seg = Mask2FormerSeg(device="cuda" if torch.cuda.is_available() else "cpu")
    masks = []
    bs = 8
    for j in tqdm(range(0, len(frames), bs), desc="Segmenting sky"):
        imgs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[j : j + bs]]
        masks.extend(seg(imgs))
    refiner = RefineMask(cascade=True)
    refined = []
    for m, f in tqdm(zip(masks, frames, strict=True), total=len(frames), desc="Refining masks"):
        refiner.mask = m
        refined.append(refiner.refine(f))
    processed = replace_sky_sequence(
        frames, sky, refined, init_orientation=(90, 90, 0), fov=50
    )
    h, w = processed[0].shape[:2]
    out = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for f in tqdm(processed, desc="Writing video"):
        out.write(f)
    out.release()
    print(f"[✓] Saved: {output_path.resolve()}")


if __name__ == "__main__":
    process_video_sky_replacement(
        video_path="./res/videos/City_ground.mp4",
        sky_image_path="./res/sky/kloofendal_48d_partly_cloudy_puresky_4k_eq.png",
        output_path="./res/out/City_ground.mp4",
        scale=0.5,
    )
