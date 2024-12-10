<?php

require_once __DIR__ . '/../../vendor/autoload.php';

use Mathis\RubixPhp\DatasetLoader;
use Mathis\RubixPhp\ModelTrainer;

$loader = new DatasetLoader();
$trainer = new ModelTrainer();

echo "Loading training dataset...\n";
$trainingDataset = $loader->loadDataset(__DIR__ . '/../../image/training');
echo "Training dataset loaded.\n";

echo "Training the model...\n";
$trainer->train($trainingDataset);
echo "Model trained.\n";

// Create models directory if it doesn't exist
$modelsDir = __DIR__ . '/../../models';
if (!is_dir($modelsDir)) {
    mkdir($modelsDir, 0777, true);
}

// Save the trained model
$trainer->saveModel($modelsDir . '/model.rbx');