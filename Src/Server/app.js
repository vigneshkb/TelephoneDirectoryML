'use strict'
const bodyParser = require('body-parser');
const express = require('express');
const ejs = require('ejs');
var fs = require('fs');


const app = express();
app.engine('html', ejs.renderFile);
app.set('view engine', 'html');

app.use(bodyParser.json({
    limit: '50mb',
    extended: true
}))
app.use(bodyParser.urlencoded({
    limit: '50mb',
    extended: true
}))
app.use(require('./routes/main'));


app.listen(9000, () => {
    console.log('App listening on port 9000');
});