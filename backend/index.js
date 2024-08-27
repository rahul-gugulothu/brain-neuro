const express = require('express');
const app = express();
const mongoose = require('mongoose');
const cors = require('cors');
const dotenv = require("dotenv");
const allroutes = require('./allroutes');
dotenv.config();
app.use(express.json());
app.use(cors({ origin: "http://localhost:3000" }));

app.use((req, res, next) => {
    console.log("Request received at " + (new Date())); 
    next();
});

const db = async () => { 
    try { 
        await mongoose.connect(process.env.DBURI, {
            useNewUrlParser: true,
            useUnifiedTopology: true,
        });
        console.log("Connected to database");
    } catch (err) {
        console.log('Error connecting to the database:', err);
    }
};
db();

// Use routes
app.use('/', allroutes);
// app.use('/',k);
const PORT = process.env.PORT || 3100;
app.listen(PORT, () => {
    console.log(`Backend server listening at port ${PORT}`);
});
