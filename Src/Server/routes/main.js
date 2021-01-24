'use strict'
const express = require('express');

var fs = require('fs');

const router = express.Router();

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

  res.redirect('/result');
});

module.exports = router;