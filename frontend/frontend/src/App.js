
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import './App.css';
import LoginRegistration from './Compounds/LoginRegistration/LoginRegistration';
import Otp from './Compounds/LoginRegistration/otp';
import HomePage from './Compounds/LoginRegistration/HomePage';

function App() {
  return (
    <div className="App">
       <Router>
        <Routes>
          <Route path="/" element={<LoginRegistration />} />
          <Route path="/otp" element={<Otp />} /> 
          <Route path="/home" element={<HomePage/>}/>

        </Routes>  
      </Router>
    </div>
  );
}

export default App;