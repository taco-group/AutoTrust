import torch
import math
import ast
import torch.nn.functional as F

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (height, width).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or \
           (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit

def resize_and_pad_image_torch(image_tensor, target_resolution):
    """
    Resize and pad an image tensor to a target resolution while maintaining aspect ratio.

    Args:
        image_tensor (torch.Tensor): The input image tensor of shape (C, H, W).
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        torch.Tensor: The resized and padded image tensor.
    """
    C, H, W = image_tensor.shape
    target_width, target_height = target_resolution

    scale_w = target_width / W
    scale_h = target_height / H

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(int(H * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(int(W * scale_h), target_width)
    resized_image = F.interpolate(image_tensor.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)
    new_image = image_tensor.new_full((C, target_height, target_width), 0.0)  # Assuming background color is zero
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    new_image[:, paste_y:paste_y + new_height, paste_x:paste_x + new_width] = resized_image
    
    # top = new_image[:, :paste_y, :]
    # middle_left = new_image[:, paste_y:paste_y + new_height, :paste_x]
    # middle_right = new_image[:, paste_y:paste_y + new_height, paste_x + new_width:]
    # bottom = new_image[:, paste_y + new_height:, :]

    # middle = torch.cat((middle_left, resized_image, middle_right), dim=2)

    # new_image = torch.cat((top, middle, bottom), dim=1)


    return new_image

def divide_to_patches_torch(image_tensor, patch_size):
    """
    Divides an image tensor into patches of a specified size.

    Args:
        image_tensor (torch.Tensor): The input image tensor of shape (C, H, W).
        patch_size (int): The size of each patch.

    Returns:
        list: A list of torch.Tensor objects representing the patches.
    """
    C, H, W = image_tensor.shape
    patches = []

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = image_tensor[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return patches

def process_anyres_image_torch(image_tensor, processor, grid_pinpoints):
    """
    Process an image tensor with variable resolutions.

    Args:
        image_tensor (torch.Tensor): The input image tensor to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if isinstance(grid_pinpoints, list):
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)

    best_resolution = select_best_resolution(image_tensor.shape[-2:], possible_resolutions)
    image_padded = resize_and_pad_image_torch(image_tensor, best_resolution)
    patches = divide_to_patches_torch(image_padded, processor.crop_size['height'])

    # Resize the original image
    shortest_edge = processor.size['shortest_edge']
    C, H, W = image_tensor.shape
    if H <= W:
        new_height = shortest_edge
        new_width = int(W * shortest_edge / H)
    else:
        new_width = shortest_edge
        new_height = int(H * shortest_edge / W)
    image_original_resize = F.interpolate(image_tensor.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)

    image_patches = [image_original_resize] + patches
    image_patches = [preprocess_image(patch, processor) for patch in image_patches]

    return torch.stack(image_patches, dim=0)

def expand2square_torch(image_tensor, background_color):
    """
    Expand an image tensor to a square shape by padding with the background color.

    Args:
        image_tensor (torch.Tensor): The input image tensor of shape (C, H, W).
        background_color (tuple): A tuple representing the background color for each channel.

    Returns:
        torch.Tensor: The expanded square image tensor.
    """
    C, H, W = image_tensor.shape
    max_side = max(H, W)
    background = image_tensor.new_full((C, max_side, max_side), 0.0)
    for c in range(C):
        background[c, :, :] = background_color[c]
    pad_top = (max_side - H) // 2
    pad_left = (max_side - W) // 2
    background[:, pad_top:pad_top+H, pad_left:pad_left+W] = image_tensor
    return background

def process_images(images, image_processor, model_cfg):
    """
    Processes a batch of images with the specified aspect ratio handling.

    Args:
        images (torch.Tensor): A tensor of images with shape (N, C, H, W).
        image_processor: The image processor object.
        model_cfg: The model configuration object.

    Returns:
        torch.Tensor: A tensor containing the processed images.
    """
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            background_color = tuple(image_processor.image_mean)
            image = expand2square_torch(image, background_color)
            image = preprocess_image(image, image_processor)
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        for image in images:
            image = process_anyres_image_torch(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        return preprocess_image(images, image_processor)
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def preprocess_image(image, image_processor):
    """
    Normalize the image from [0,1] to N(0,1)

    Args:
        image (torch.Tensor): Image tensor

    Returns:
        torch.Tensor: Normalized image tensor
    """
    image_mean = torch.tensor(image_processor.image_mean).half().to(image.device)
    image_std = torch.tensor(image_processor.image_std).half().to(image.device)
    image = (image - image_mean[None,:,None,None]) / image_std[None,:,None,None]
    return F.interpolate(image, size=(336, 336), mode='bilinear', align_corners=False)