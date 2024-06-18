def add_color_hints(grayscale_img, original_img, num_hints=10):
    grayscale_arr = np.array(grayscale_img)
    original_arr = np.array(original_img)

    if len(grayscale_arr.shape) == 2:
        grayscale_arr = np.stack((grayscale_arr,) * 3, axis=-1)  # Make it a 3-channel image

    # Add color hints from the original image
    for _ in range(num_hints):
        y, x = random.randint(0, grayscale_arr.shape[0] - 1), random.randint(0, grayscale_arr.shape[1] - 1)
        color_hint = original_arr[y, x]  # Get a color from the same position in the original image
        grayscale_arr[y, x] = color_hint  # Apply the color hint at the position

    # Convert the numpy array back to a PIL image
    hinted_img = Image.fromarray(grayscale_arr.astype('uint8'), 'RGB')
    return hinted_img

def create_random_hinted_images(base_dir, output_dir_suffix):
    print(f"Processing directory: {base_dir}")
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            output_dir = dir_path.replace(base_dir, base_dir + output_dir_suffix)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")

            for filename in os.listdir(dir_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(dir_path, filename)
                    try:
                        color_img = Image.open(img_path).convert('RGB')
                        grayscale_img = Image.open(img_path).convert('L')
                        hinted_img = add_color_hints(grayscale_img, color_img)
                        hinted_img.save(os.path.join(output_dir, filename))
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")

dataset_dirs = {
    'train': '/content/CINIC-10/train',
    'valid': '/content/CINIC-10/valid',
    'test': '/content/CINIC-10/test'
}

for ds_name, ds_path in dataset_dirs.items():
    create_random_hinted_images(ds_path, f'_hints')

class CINIC10DatasetWithHints(Dataset):
    def __init__(self, original_dir, hints_dir, transform=None):
        super(CINIC10DatasetWithHints, self).__init__()
        self.original_dir = original_dir
        self.hints_dir = hints_dir
        self.transform = transform
        self.images = []  # List to hold paths to the images

        # Loop through each subdirectory in the original directory
        for category in os.listdir(original_dir):
            category_dir = os.path.join(original_dir, category)
            if os.path.isdir(category_dir):  # Check if it is a directory
                for img_file in os.listdir(category_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(category, img_file))  # Save relative path

        print(f"Loaded {len(self.images)} images from {original_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        if idx >= len(self.images):
             raise IndexError(f"Requested index {idx} exceeds dataset size {len(self.images)}")

        img_name = self.images[idx]
        original_img_path = os.path.join(self.original_dir, img_name)
        hinted_img_path = os.path.join(self.hints_dir, img_name)

        original_img = Image.open(original_img_path).convert('RGB')
        hinted_img = Image.open(hinted_img_path).convert('RGB')

        if self.transform:
            original_img = self.transform(original_img)
            hinted_img = self.transform(hinted_img)

        return hinted_img, original_img