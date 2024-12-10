<?php

namespace Mathis\RubixPhp;

use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Datasets\Labeled;

class ModelTrainer
{
    private $estimator;

    public function __construct()
    {
        $this->estimator = new ClassificationTree();
    }

    public function train(Labeled $trainingDataset): void
    {
        $vectorizer = new ImageVectorizer();
        $trainingDataset->apply($vectorizer);
        $this->estimator->train($trainingDataset);
    }

    public function test(Labeled $testingDataset): float
    {
        $vectorizer = new ImageVectorizer();
        $testingDataset->apply($vectorizer);
        $predictions = $this->estimator->predict($testingDataset);
        $metric = new Accuracy();
        return $metric->score($predictions, $testingDataset->labels());
    }

    public function saveModel(string $filePath): void
    {
        file_put_contents($filePath, serialize($this->estimator));
    }

    public function loadModel(string $filePath): void
    {
        $this->estimator = unserialize(file_get_contents($filePath));
    }
}