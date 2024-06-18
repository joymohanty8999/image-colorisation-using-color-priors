def display_hinted_and_original_images(dataloader, num_batches=1, num_images_per_batch=5):

    batch_count = 0

    for hinted_images, original_images in dataloader:
        plt.figure(figsize=(10, 2 * num_images_per_batch))

        # Loop through each image in the batch, up to the specified number of images
        for i in range(min(num_images_per_batch, hinted_images.shape[0])):
            # Display hinted image
            ax = plt.subplot(2, num_images_per_batch, i + 1)
            plt.imshow(hinted_images[i].permute(1, 2, 0))
            ax.set_title("Hinted Image")
            plt.axis('off')

            # Display original image
            ax = plt.subplot(2, num_images_per_batch, num_images_per_batch + i + 1)
            plt.imshow(original_images[i].permute(1, 2, 0))
            ax.set_title("Original Image")
            plt.axis('off')

        plt.show()

        batch_count += 1
        if batch_count >= num_batches:
            break
        
display_hinted_and_original_images(train_loader, num_batches=1, num_images_per_batch=5)