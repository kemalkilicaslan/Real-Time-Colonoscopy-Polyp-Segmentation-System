import numpy as np

from polypseg.overlays import device_overlay_mask


def mucosa():
    img = np.zeros((288, 384, 3), np.uint8)
    img[...] = (190, 90, 80)  # pinkish red
    return img


def test_teal_scopeguide_box_is_masked_with_padding():
    img = mucosa()
    img[200:270, 10:110] = (20, 165, 137)  # colour measured on HyperKvasir video
    m = device_overlay_mask(img, pad=4)
    assert m[200:270, 10:110].all()
    assert m[196, 50] and not m[190, 50]
    assert not m[:150].any()


def test_no_overlay_on_plain_mucosa_or_small_green_specks():
    img = mucosa()
    img[50:53, 50:53] = (20, 165, 137)  # tiny speck, not a UI box
    assert not device_overlay_mask(img).any()


def test_ignore_mask_covers_area_outside_field_of_view():
    import cv2

    from polypseg.overlays import ignore_mask

    frame = np.zeros((576, 720, 3), np.uint8)
    cv2.circle(frame, (400, 288), 250, (190, 90, 80), -1)
    cv2.putText(frame, "Sex: Age:", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,) * 3, 2)
    m = ignore_mask(frame)
    assert m[30, 40]  # burnt-in text on the black frame
    assert not m[288, 400]  # centre of the field of view
