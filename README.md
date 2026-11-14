# 🍽️ Moroccan Food Classifier (70 dishes)

An educational project that classifies images of Moroccan dishes using transfer learning (MobileNetV2).

## Results (on a held-out test set the model never saw)
- Accuracy: 82.5%
- Correct dish within the top 3 predictions: 93.6%
- Split: train 6012 / validation 1296 / test 1291 (near-duplicate images were kept together in the same split)

For reference, random guessing over 70 classes gives about 1.4%.

## Weaknesses
- Dishes with similar-looking sauces (for example liver with sauce vs. meatballs with tomato sauce).
- Classes with very few images, whose test results are not statistically stable.

## Data
MFOOD_70 on Kaggle (license: CC BY-SA 4.0). The images are not included in this repository; please download them from the original source and credit the dataset's authors.

## Method
1. Remove corrupted images and group near-duplicates (perceptual hashing).
2. Split by group into train / validation / test.
3. Train in two stages: a new classification head on a frozen MobileNetV2, then fine-tuning of the last layers, with class weights to handle imbalance.
4. Evaluate once on the test set.

## Disclaimer
Educational project. The model only knows 70 dishes, so for any other image it will pick the closest of those 70 and may be wrong.

## Author
Hamza Oulbayyad
