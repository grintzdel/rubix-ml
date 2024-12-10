<?php

require 'vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

// Function to load images and their labels
function loadDataset(string $directory): Labeled
{
    $samples = [];
    $labels = [];

    foreach (range(0, 9) as $label) {
        $labelDir = $directory . '/' . $label;
        $images = glob($labelDir . '/*.png');

        foreach ($images as $imagePath) {
            $image = imagecreatefrompng($imagePath);
            imagefilter($image, IMG_FILTER_GRAYSCALE);

            $width = imagesx($image);
            $height = imagesy($image);

            $pixels = [];
            for ($y = 0; $y < $height; $y++) {
                for ($x = 0; $x < $width; $x++) {
                    $colorIndex = imagecolorat($image, $x, $y);
                    $color = imagecolorsforindex($image, $colorIndex);
                    $pixels[] = $color['red']; // Since the image is grayscale, red, green, and blue values are the same
                }
            }

            $samples[] = $pixels;
            $labels[] = 'digit_' . $label;

            imagedestroy($image);
        }
    }

    return Labeled::build($samples, $labels);
}

echo "Loading training dataset...\n";
$trainingDataset = loadDataset('image/training');
echo "Training dataset loaded.\n";

echo "Loading testing dataset...\n";
$testingDataset = loadDataset('image/testing');
echo "Testing dataset loaded.\n";

// Transform the images
$vectorizer = new ImageVectorizer();

echo "Transforming training dataset...\n";
$trainingDataset->apply($vectorizer);
echo "Training dataset transformed.\n";

echo "Transforming testing dataset...\n";
$testingDataset->apply($vectorizer);
echo "Testing dataset transformed.\n";

// Create and train the model
$estimator = new ClassificationTree();
echo "Training the model...\n";
$estimator->train($trainingDataset);
echo "Model trained.\n";

// Evaluate the model
echo "Testing the model...\n";
$predictions = $estimator->predict($testingDataset);
$metric = new Accuracy();
$accuracy = $metric->score($predictions, $testingDataset->labels());

echo 'Accuracy: ' . ($accuracy * 100) . "%\n";