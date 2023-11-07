const express = require('express');
const multer = require('multer');
const bodyParser = require('body-parser');
const app = express();
const cors = require('cors');
const base64_decode = require('node-base64-image').decode;
const { PythonShell } = require('python-shell')

app.use(bodyParser.json());

app.use(cors());

const ScriptPath = String.raw`D:\works\SignProj\CentralPonts`;
const PythonPath = String.raw`D:\Python\Python311\python.exe`;
const ServerPath = String.raw`D:\works\SignProj\Server`;

const ScriptPathGAN = String.raw`D:\works\SignProj\GAN-Model\CudaProject`;
const PythonPathGAN = String.raw`D:\works\SignProj\GAN-Model\CudaProject\TorchEnv\TorchEnv\env\Scripts\python.exe`;



app.get('/', function (request, response) {
    response.end();
});


const upload = multer({
    dest: 'files/',
    limits: { fieldSize: 25 * 1024 * 1024 }
});

app.post('/TrainDots', function (request, response) {
    const fs = require('fs');
    var temp = request.body;
    console.log(request.body);

    //flip the rows by the order of appear
    for (var i = 0; i < 10; i++) {
        const arr = [...temp[i]];
        const reverse_arr = arr.reverse();
        temp[i] = reverse_arr;
    }

    for (var i = 0; i < 10; i++) {
        console.log(temp[i]);
        console.log(" ");
    }

    const path = require('path');

    // Read the list of files in the folder
    fs.readdir(ServerPath, (err, files) => {
        if (err) {
            console.error('Error reading folder:', err);
            return;
        }

        // Filter for CSV files
        const csvFiles = files.filter(file => file.endsWith('.csv'));

        // Delete each CSV file
        csvFiles.forEach(csvFile => {
            const filePath = path.join(ServerPath, csvFile);
            fs.unlinkSync(filePath); // Use unlinkSync for synchronous deletion
            console.log(`${csvFile} deleted successfully.`);
        });

        // Create new starCSV files
        temp.forEach((arr, index) => {
            const csvData = arr.map(coord => `${coord.x},${coord.y}`).join('\n');
            const fileName = path.join(ServerPath, `star_${index}.csv`);

            fs.writeFileSync(fileName, csvData);

            console.log(`CSV file "star_${fileName}" created.`);
        });

        // Create new CSV files
        temp.forEach((arr, index) => {
            // Filter out coordinates with x and y as "9999"
            const filteredCoords = arr.filter(coord => coord.x !== "9999" && coord.y !== "9999");

            // Convert the filtered coordinates to CSV data
            const csvData = filteredCoords.map(coord => `${coord.x},${coord.y}`).join('\n');
            const fileName = path.join(ServerPath, `${index}.csv`);

            fs.writeFileSync(fileName, csvData);

            console.log(`CSV file "${fileName}" created.`);
        });
    });
    ////////////////////////// run the training script
    const options = {
        mode: 'text',
        scriptPath: ScriptPath,
        pythonPath: PythonPath,
        pythonOptions: ['-u'], // get print results in real-time
    };
    PythonShell.run('Run_Register.py', options).then(messages => {
        // Get the desired result from messages[2]
        const result = messages[messages.length - 1];      
        console.log(result);

        // Send the result back to the client
        response.send(result);
    }).catch(error => {
        console.error(error);
        response.status(500).send('An error occurred');
    });



});

//////////////

app.post('/GAN', function (request, response) {
    const fs = require('fs');
    console.log("GAN Model..");
    
    //run the training script
    const options = {
        mode: 'text',
        scriptPath: ScriptPathGAN,
        pythonPath: PythonPathGAN,
        pythonOptions: ['-u'], // get print results in real-time
    };
    PythonShell.run('GAN_main_train.py', options).then(messages => {
        // Get the desired result from messages[2]
        const result = messages[messages.length - 1];
        console.log(result);
        console.log("GAN model train is done");
        // Send the result back to the client
        response.send(result);
    }).catch(error => {
        console.error(error);
        response.status(500).send('An error occurred');
    });

});
///////////////////////////

app.post('/LoginDots', function (request, response) {
    const fs = require('fs');
    var temp = request.body;


    const arrR = temp[0];
    const reverse_arr = arrR.reverse();
    temp[0] = reverse_arr;

    console.log(temp[0]);
    console.log(" ");

    temp.forEach((arr, index) => {
        const csvData = arr.map(coord => `${coord.x},${coord.y}`).join('\n');
        const fileName = ServerPath + '\\star_login.csv';

        fs.writeFileSync(fileName, csvData);

        console.log(`CSV file "star_login" created.`);
    });

    temp.forEach((arr, index) => {
        // Filter out coordinates with x or y as "9999"
        const filteredCoords = arr.filter(coord => coord.x !== "9999" && coord.y !== "9999");

        if (filteredCoords.length > 0) {
            // Convert the filtered coordinates to CSV data
            const csvData = filteredCoords.map(coord => `${coord.x},${coord.y}`).join('\n');
            const fileName = ServerPath + '/login.csv';

            fs.writeFileSync(fileName, csvData);

            console.log(`CSV file "login" created.`);
        }
    });
    ////////////////////////// run the login script
    var res;
    const options = {
        mode: 'text',
        scriptPath: ScriptPath,
        pythonPath: PythonPath,
        pythonOptions: ['-u'], // get print results in real-time
    };
    PythonShell.run('Run_Login.py', options).then(messages => {
        // Get the desired result from messages[2]
        const result = messages[messages.length - 1];
        console.log(result);
        res = result;
        // Send the result back to the client
        response.send(result);
    }).catch(error => {
        console.error(error);
        response.status(500).send('An error occurred');
    });

});


const port = process.env.PORT || 1337;
app.listen(port, function () {
    console.log("Server running on port: " + port);
});