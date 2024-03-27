import cv2

def compare_histograms(image1_path, image2_path):
    # Read the images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    img1 = cv2.resize(src=img1, dsize=(256, 256))
    img2 = cv2.resize(src=img2, dsize=(256, 256))

    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate histograms
    hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])

    # Normalize histograms
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # Calculate Bhattacharyya distance
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    return similarity

def main():
    # Replace these paths with the paths to your images
    image1_path = r'CHNCXR_0001_0.jpg'
    image2_path = r'lag.jpg'

    # Compare histograms
    similarity = compare_histograms(image1_path, image2_path)

    # Set a threshold for similarity (e.g., 80%)
    threshold = 0.35

    print("Similarity value: ", similarity)

    if similarity <= threshold:
        print(f"The histograms are {similarity * 100:.2f}% similar. Do some work.")
    else:
        print(f"The histograms are {similarity * 100:.2f}% dissimilar. Do some other work.")
        # Your code for the case where similarity is less than the threshold

if __name__ == "__main__":
    main()
