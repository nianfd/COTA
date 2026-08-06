import numpy as np
from subspacead.config import parse_layer_indices, parse_grouped_layers
import logging
import cv2
from subspacead.core.extractor import FeatureExtractor
from subspacead.core.otm import calculate_task_scores
from subspacead.post_process.scoring import post_process_map


def get_patch_coords(image_height, image_width, patch_size, overlap):
    """Calculates patch coordinates with edge-case handling."""
    coords = []
    stride = int(patch_size * (1 - overlap))
    for y in range(0, image_height, stride):
        for x in range(0, image_width, stride):
            x1, y1 = x, y
            x2, y2 = min(x + patch_size, image_width), min(y + patch_size, image_height)
            # Ensure patches at the edges are full size
            if (x2 - x1) < patch_size or (y2 - y1) < patch_size:
                x1, y1 = max(0, x2 - patch_size), max(0, y2 - patch_size)
            coords.append((x1, y1, x2, y2))
    return coords


def _get_patch_background_mask(
    saliency_masks_batch, threshold_method, percentile_threshold
):
    """Applies thresholding to a batch of saliency masks."""
    background_mask = np.zeros_like(saliency_masks_batch, dtype=bool)
    for j, saliency_map in enumerate(saliency_masks_batch):
        try:
            if threshold_method == "percentile":
                threshold = np.percentile(saliency_map, percentile_threshold * 100)
                background_mask[j] = saliency_map < threshold
            else:  # 'otsu'
                norm_mask = cv2.normalize(
                    saliency_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                _, binary_mask = cv2.threshold(
                    norm_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                background_mask[j] = binary_mask == 0
        except Exception as e:
            logging.warning(f"Saliency mask failed for patch {j}: {e}. Skipping mask.")
    return background_mask


def _stitch_patches_to_canvas(
    full_map_canvas, count_map_canvas, patches_to_stitch, coord_batch, blur=True
):
    """Stitches a batch of processed patches onto their respective canvases."""
    for j, patch_map in enumerate(patches_to_stitch):
        x1, y1, x2, y2 = coord_batch[j]
        patch_h, patch_w = y2 - y1, x2 - x1

        map_patch_resized = post_process_map(patch_map, (patch_h, patch_w), blur=blur)

        full_map_canvas[y1:y2, x1:x2] += map_patch_resized
        count_map_canvas[y1:y2, x1:x2] += 1


def _process_single_image_patched(
    pil_img,
    extractor: FeatureExtractor,
    pca_params,
    args,
    device,
    h_p,
    w_p,
    feature_dim,
    otm_params=None,
    source_id=None,
):
    """Return stitched image-branch, pixel-branch, and saliency maps."""
    img_width, img_height = pil_img.size
    patch_coords = get_patch_coords(
        img_height, img_width, args.patch_size, args.patch_overlap
    )

    # Create canvases for stitching
    image_map_full = np.zeros((img_height, img_width), dtype=np.float32)
    image_count_map = np.zeros((img_height, img_width), dtype=np.float32)
    pixel_map_full = np.zeros((img_height, img_width), dtype=np.float32)
    pixel_count_map = np.zeros((img_height, img_width), dtype=np.float32)
    saliency_map_full = np.zeros((img_height, img_width), dtype=np.float32)
    s_count_map = np.zeros((img_height, img_width), dtype=np.float32)

    # Parse layer arguments once
    layers = parse_layer_indices(args.layers)
    grouped_layers = (
        parse_grouped_layers(args.grouped_layers) if args.agg_method == "group" else []
    )

    # Process patches in batches
    for i in range(0, len(patch_coords), args.batch_size):
        coord_batch = patch_coords[i : i + args.batch_size]
        patch_batch = [pil_img.crop(c) for c in coord_batch]

        tokens, _, saliency_masks_batch = extractor.extract_tokens(
            patch_batch,
            args.image_res,
            layers,
            args.agg_method,
            grouped_layers,
            args.docrop,
            use_clahe=args.use_clahe,
            dino_saliency_layer=args.dino_saliency_layer,
        )

        query_source_ids = None
        if source_id is not None:
            query_source_ids = np.full(
                tokens.reshape(-1, feature_dim).shape[0],
                int(source_id),
                dtype=np.int64,
            )

        image_scores, pixel_scores = calculate_task_scores(
            tokens.reshape(-1, feature_dim),
            pca_params,
            otm_params,
            args.otm_variant,
            score_method=args.score_method,
            drop_k=args.drop_k,
            query_source_ids=query_source_ids,
            bank_chunk=args.otm_bank_chunk,
        )
        image_maps_batch = image_scores.reshape(len(patch_batch), h_p, w_p)
        pixel_maps_batch = pixel_scores.reshape(len(patch_batch), h_p, w_p)

        if args.bg_mask_method is not None:
            background_mask = _get_patch_background_mask(
                saliency_masks_batch,
                args.mask_threshold_method,
                args.percentile_threshold,
            )
            image_maps_batch[background_mask] = 0.0
            pixel_maps_batch[background_mask] = 0.0

        # Stitch image- and pixel-task maps independently.
        _stitch_patches_to_canvas(
            image_map_full,
            image_count_map,
            image_maps_batch,
            coord_batch,
            blur=True,
        )
        _stitch_patches_to_canvas(
            pixel_map_full,
            pixel_count_map,
            pixel_maps_batch,
            coord_batch,
            blur=True,
        )

        # Stitch saliency maps
        _stitch_patches_to_canvas(
            saliency_map_full,
            s_count_map,
            saliency_masks_batch,
            coord_batch,
            blur=False,
        )

    # Average the scores in overlapping regions
    image_map_final = np.divide(
        image_map_full,
        image_count_map,
        out=np.zeros_like(image_map_full),
        where=image_count_map != 0,
    )
    pixel_map_final = np.divide(
        pixel_map_full,
        pixel_count_map,
        out=np.zeros_like(pixel_map_full),
        where=pixel_count_map != 0,
    )
    saliency_map_final = np.divide(
        saliency_map_full,
        s_count_map,
        out=np.zeros_like(saliency_map_full),
        where=s_count_map != 0,
    )

    return image_map_final, pixel_map_final, saliency_map_final


def process_image_patched(
    pil_imgs: list,
    extractor: FeatureExtractor,
    pca_params,
    args,
    h_p,
    w_p,
    feature_dim,
    otm_params=None,
    source_ids=None,
):
    """Return image-task maps, pixel-task maps, and saliency maps."""
    image_maps_final = []
    pixel_maps_final = []
    saliency_maps_final = []

    if args.bg_mask_method == "pca_normality":
        logging.warning(
            "Patching mode is not compatible with 'pca_normality' masking. "
            "Falling back to 'dino_saliency' if masking is enabled."
        )

    if source_ids is not None and len(source_ids) != len(pil_imgs):
        raise ValueError("source_ids must contain one identifier per input image.")

    for image_index, pil_img in enumerate(pil_imgs):
        source_id = None if source_ids is None else source_ids[image_index]
        image_map, pixel_map, saliency_map = _process_single_image_patched(
            pil_img,
            extractor,
            pca_params,
            args,
            h_p,
            w_p,
            feature_dim,
            otm_params=otm_params,
            source_id=source_id,
        )
        image_maps_final.append(image_map)
        pixel_maps_final.append(pixel_map)
        saliency_maps_final.append(saliency_map)

    return image_maps_final, pixel_maps_final, saliency_maps_final
