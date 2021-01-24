'use strict'
const express = require('express');

var fs = require('fs');

const router = express.Router();

const { exec } = require("child_process");

//Render home page
router.get('/', (req, res) => {
  res.render('Home.html');
});

router.get('/result', (req, res) => {
  console.log('Redirecting...');
  res.render('Result.html');
});


router.post('/upload', function (req, res) {
  console.log('image received');
  var image = req.body.imgData;
  var data = image.replace(/^data:image\/\w+;base64,/, '');

  fs.writeFile('./../../Data/Input/snap.png', data, {
    encoding: 'base64'
  }, function (err) {
  });
  console.log('******** File created from base64 encoded string ********');

  exec("python ./../Scripts/main.py --validate", (error, stdout, stderr) => {
    if (error) {
      console.log(`error: ${error.message}`);
    }
    if (stderr) {
      console.log(`stderr: ${stderr}`);
    }
    console.log(`stdout: ${stdout}`);
    res.json(JSON.parse(stdout));
  });
});

module.exports = router;