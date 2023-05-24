import React, { useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Button, LinearProgress, Typography } from "@mui/material";

const PredictionComponent = ({link, decease}) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictionResult, setPredictionResult] = useState('');
  const [predictionResultList, setPredictionResultList] = useState('');

  const loadImage = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          resolve(img);
        };
        img.onerror = (error) => {
          reject(error);
        };
        img.src = event.target.result;
      };
      reader.onerror = (error) => {
        reject(error);
      };
      reader.readAsDataURL(file);
    });
  };


  const [loading, setLoading] = useState(false);


  const loadModel = async () => {
    const model = await tf.loadLayersModel(link);
    performPrediction(model);
  };

  const performPrediction = async (model) => {
    setLoading(true);
    const img = await loadImage(selectedImage);
    setLoading(false);
    const tensorImg = tf.browser.fromPixels(img);
    const resizedImg = tf.image.resizeBilinear(tensorImg, [224, 224]);
    const processedImg = resizedImg.expandDims(0).toFloat().div(255);
    const prediction = model.predict(processedImg);
    const result = prediction.dataSync(); // Get the prediction result as an array
    console.log(result);
    setPredictionResult(`Class1: ${result[0]}, Class2 ${result[1]}`); // Update the state with the prediction result
    setPredictionResultList(result)
    tf.dispose([img, resizedImg, processedImg, prediction]);
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setSelectedImage(file);
  };



  return (
    <div>
        {loading && (
          <LinearProgress
            sx={{
              position: "absolute",
              top: { xs: "58px", sm: "64px", md: "64px" },
              left: 0,
              right: 0,
              zIndex: 1,
            }}
          />
        )}

      <input type="file" accept="image/*" onChange={handleImageUpload} />
      <Button variant="contained" color="primary" onClick={loadModel}>
        Predict
      </Button>
      {predictionResult && (
        <Typography variant="h6"
        sx={{marginTop: "20px"}}
        >
          {predictionResult}
        </Typography>
      )}

        {predictionResult && (
        <Typography variant="h4"
        sx={{marginTop: "20px"}}
        >
          Prediction Result: {predictionResultList[0] < predictionResultList[1] ? `${decease}` : `No ${decease}`}
        </Typography>
      )}
    </div>
  );
};

export default PredictionComponent;
