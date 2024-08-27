import React, { useState } from 'react';
import axios from 'axios';
import './HomePage.css';

const Predict = () => {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [resultImageUrl, setResultImageUrl] = useState('');
  const [error, setError] = useState('');

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    setPrediction('');
    setResultImageUrl('');
    setError('');

    if (selectedFile) {
      const imageUrl = URL.createObjectURL(selectedFile);
      setImageUrl(imageUrl);
    } else {
      setImageUrl('');
    }
  };

  const handleSubmitDLClassification = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data.prediction);
      setResultImageUrl('');
    } catch (error) {
      console.error('Error uploading the file:', error);
      setError('Error uploading the file');
    }
  };

  const handleSubmitDLSegmentation = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('http://localhost:5002/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob', 
      });
      const resultImageUrl = URL.createObjectURL(response.data);
      setResultImageUrl(resultImageUrl);
      setPrediction(''); 
    } catch (error) {
      console.error('Error uploading the file:', error);
      setError('Error uploading the file');
    }
  };

  const handleSubmitLLMSegmentation = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('http://localhost:5001/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob', 
      });
      const resultImageUrl = URL.createObjectURL(response.data);
      setResultImageUrl(resultImageUrl);
      setPrediction(''); 
    } catch (error) {
      console.error('Error uploading the file:', error);
      setError('Error uploading the file');
    }
  };

  const handleSubmitLLMClassification = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('http://localhost:5003/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data.prediction);
      setResultImageUrl(''); 
    } catch (error) {
      console.error('Error uploading the file:', error);
      setError('Error uploading the file');
    }
  };

  return (
    <div className="App">
      <h1>Upload an image to classify or segment</h1>

      <form onSubmit={handleSubmitDLClassification}>
        <div className="input-container">
          <input className='input' type="file" onChange={handleFileChange} />
          {imageUrl && (
            <div>
              <img src={imageUrl} alt="Uploaded" className="uploaded-image" />
            </div>
          )}
        </div>
        <button className='button' type="submit">DL classification</button>
      </form>

      <form onSubmit={handleSubmitDLSegmentation}>
        <button className='button' type="submit">DL segmentation</button>
      </form>

      <form onSubmit={handleSubmitLLMSegmentation}>
        <button className='button' type="submit">LLM segmentation</button>
      </form>

      <form onSubmit={handleSubmitLLMClassification}>
        <button className='button' type="submit">LLM classification</button>
      </form>

      {resultImageUrl && (
        <div className="result-container">
          <h2>Segmentation Result:</h2>
          <img src={resultImageUrl} alt="Segmentation Result" className="result-image" />
        </div>
      )}

      {prediction && <div>Prediction: {prediction}</div>}
      {error && <div className='error'>{error}</div>}
    </div>
  );
};

export default Predict;
