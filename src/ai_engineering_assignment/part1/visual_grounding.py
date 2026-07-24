from PIL import ImageDraw
from PIL import Image


def draw_multiple_bboxes(
    doc,
    items,
    ref_lookup,
    selected_ref=None,
    highlight_alpha=90,
):
    """Draw bounding boxes on the rendered image for selected chunks"""
    page_no = items[0].prov[0].page_no
    page = doc.pages[page_no]
    pil_image = page.image.pil_image.copy().convert("RGBA")

    scale_x = pil_image.width / page.size.width
    scale_y = pil_image.height / page.size.height

    # Transparent overlay for the yellow highlight fill
    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for item in items:
        bbox = item.prov[0].bbox.to_top_left_origin(page_height=page.size.height)
        left, top, right, bottom = (
            bbox.l * scale_x,
            bbox.t * scale_y,
            bbox.r * scale_x,
            bbox.b * scale_y,
        )

        # Semi-transparent yellow fill (highlighter effect)
        overlay_draw.rectangle(
            [(left, top), (right, bottom)], fill=(255, 255, 0, highlight_alpha)
        )

    # Composite the highlight overlay onto the base image
    pil_image = Image.alpha_composite(pil_image, overlay)

    # Draw solid red borders + labels on top, after compositing
    draw = ImageDraw.Draw(pil_image)
    for item in items:
        ref = ref_lookup[item.self_ref]

        # Skip all non-selected chunks
        if ref != selected_ref:
            continue

        bbox = item.prov[0].bbox.to_top_left_origin(page_height=page.size.height)

        left, top, right, bottom = (
            bbox.l * scale_x,
            bbox.t * scale_y,
            bbox.r * scale_x,
            bbox.b * scale_y,
        )

        draw.rectangle(
            [(left, top), (right, bottom)],
            outline="red",
            width=4,
        )

        draw.text(
            (left, max(top - 15, 0)),
            ref.split("/")[-1],
            fill="red",
        )
    return pil_image.convert("RGB")  # convert back for saving as JPEG/normal display
