const express = require("express");
const usersModel = require("./userModel");
const allroutes = express.Router();
const multer = require("multer");
const upload = multer();
const nodemailer = require('nodemailer');

const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: 'rahulg.321r@gmail.com',
        pass: 'qhsr hevn iisa kvyk',
    }
});

const otpStorage = new Map();

// Middleware to parse JSON and form data
allroutes.use(express.json());
allroutes.use(express.urlencoded({ extended: true }));

allroutes.get('/', (req, res) => {
    console.log("Reached root");
    res.send("Welcome to our page");
});

allroutes.post('/signUp', async (req, res) => {
    try {
        const { name, email, password } = req.body;
        console.log('Received sign-up data:', req.body);

        if (!name || !email || !password) {
            return res.status(400).json({ error: "All fields (name, email, password) are required" });
        }

        const existingUser = await usersModel.findOne({ email });
        if (existingUser) {
            return res.status(400).send({ error: "Email already exists" });
        }

        const newUser = new usersModel({
            username: name,
            email,
            password
        });

        await newUser.save();
        res.status(201).json(newUser);
    } catch (err) {
        console.error("Error while adding user:", err);
        res.status(500).send({ error: "Internal Server Error" });
    }
});

allroutes.post('/login', upload.none(), async (req, res) => {
    try {
        const { email, password } = req.body;

        if (!email || !password) {
            return res.status(400).send({ error: "Email and password are required" });
        }

        const user = await usersModel.findOne({ email });

        if (!user) {
            return res.status(401).json({ error: "Invalid credentials" });
        }

        if (user.password === password) {
            res.send(user);
            console.log("Login successful");
        } else {
            res.status(401).send({ error: "Invalid credentials" });
        }
    } catch (err) {
        console.error("Error during login:", err);
        res.status(500).send({ error: "Internal Server Error" });
    }
});

allroutes.post('/send-otp', upload.none(), async (req, res) => {
    const { email } = req.body;

    if (!email) {
        return res.status(400).send({ error: 'Email address is required' });
    }

    const otp = Math.floor(100000 + Math.random() * 900000);
    otpStorage.set(email, otp);

    const mailOptions = {
        from: 'rahulg.321r@gmail.com',
        to: email,
        subject: 'Your OTP',
        text: `Your OTP is: ${otp}`,
    };

    try {
        const info = await transporter.sendMail(mailOptions);
        console.log('Email sent:', info.response);
        res.status(200).json({ message: 'OTP sent successfully' });
    } catch (error) {
        console.error('Error sending email:', error.message);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

allroutes.post('/verify-otp', upload.none(), (req, res) => {
    const { email, userEnteredOtp } = req.body;

    if (!email || !userEnteredOtp) {
        return res.status(400).json({ error: 'Email and OTP are required' });
    }

    const storedOtp = otpStorage.get(email);

    if (storedOtp && storedOtp.toString() === userEnteredOtp) {
        otpStorage.delete(email); // Optionally, clear the OTP after successful verification
        res.status(200).json({ message: 'OTP verified successfully' });
    } else {
        res.status(400).json({ error: 'Invalid OTP' });
    }
});

module.exports = allroutes;
