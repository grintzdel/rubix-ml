<?php

require_once __DIR__ . '/../../vendor/autoload.php';

use Mathis\RubixPhp\DatasetLoader;
use Mathis\RubixPhp\ModelTrainer;

$loader = new DatasetLoader();
$trainer = new ModelTrainer();

echo "Loading testing dataset...\n";
$testingDataset = $loader->loadDataset('../../image/testing');
echo "Testing dataset loaded.\n";

// Load the trained model
$trainer->loadModel('../../models/model.rbx');

echo "Testing the model...\n";
$accuracy = $trainer->test($testingDataset);
echo 'Accuracy: ' . ($accuracy * 100) . "%\n";