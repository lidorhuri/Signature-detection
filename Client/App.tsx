import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, PanResponder, ActivityIndicator } from 'react-native';
import { I18nManager } from "react-native";
import Svg, { Path } from 'react-native-svg';

const SignaturePad = () => {
  const [currentPage, setCurrentPage] = useState('main');
  const [touchPoints, setTouchPoints] = useState([]);
  const [viewPoints, setViewPoints] = useState([]);
  const [currentSignature, setCurrentSignature] = useState(1); // Counter for signatures
  const [allSignatures, setAllSignatures] = useState([]);
  const signaturesMaxedOut = currentSignature === 10;
  const NGROK_LINK = 'https://3a24-85-250-125-167.ngrok-free.app'

    //controling the strok on the area 
    const [lastX, setLastX] = useState(null);
    const [lastY, setLastY] = useState(null);
    const PIXEL_INTERVAL = 1; // Change this to your desired pixel interval
  
  
    const [currentPath, setCurrentPath] = useState('');
    const [allPaths, setAllPaths] = useState([]);
    const [notification, setNotification] = useState('');
    const [startPoint, setStartPoint] = useState(null)
  
  
    const panResponder = PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderMove: (evt) => {
        const { nativeEvent } = evt;
        const { locationX, locationY } = nativeEvent;
  
        // Get the current touch coordinates
        const x = evt.nativeEvent.locationX;
        const y = evt.nativeEvent.locationY;
  
        // Add the new point to the path
        setCurrentPath((prevPath) => prevPath + ` L${x} ${y}`);
  
        if (lastX === null || lastY === null) {
          setLastX(locationX);
          setLastY(locationY);
        }
  
        const deltaX = locationX - lastX;
        const deltaY = locationY - lastY;
        const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
  
        if ((distance >= PIXEL_INTERVAL) && (locationX >= 0 && locationX <= 300 && locationY >= 0 && locationY <= 300) && !(locationX < 5 || locationY < 5)) {
          // Trigger your event here
          var calcX = locationX; //x for view in the pad
          //console.log('Event triggered at', locationX, locationY);
          if (I18nManager.isRTL) {
            calcX = 300 - calcX;
          }
  
          const updatedTouchPoints = [...touchPoints, { x: Math.round(locationX), y: Math.round(locationY) }]; //adding the new point to the rest of the points
          const updatedViewPoints = [...viewPoints, { x: Math.round(calcX), y: Math.round(locationY) }]; //adding the new point to the rest of the points
          setTouchPoints(updatedTouchPoints);
          setViewPoints(updatedViewPoints)
          // Update lastX and lastY
          setLastX(locationX);
          setLastY(locationY);
        }
      },
      onPanResponderGrant: (event) => {
        // Get the initial touch coordinates
        const x = event.nativeEvent.locationX;
        const y = event.nativeEvent.locationY;
        setStartPoint({ x, y });
  
        // Start a new path
        setCurrentPath(`M${x} ${y}`);
      },
      onPanResponderRelease: () => {
        // Save the completed path to the list of all paths
        const updatedTouchPoints = [...touchPoints, { x: '9999' ,y: '9999'}];
        setTouchPoints(updatedTouchPoints);

        if (currentPath) {
          setAllPaths([...allPaths, currentPath]);
        }
        // Reset the current path
        setCurrentPath('');
      },
    });
  
  
    async function handleSave() {
      if (touchPoints.length <= 8) { //
        alert("Your singature is too short!")
        setTouchPoints([]); // Reset the doodle after saving
        setViewPoints([]);
        return;
      }
      // Save the touch points for the current signature
      const updatedAllSignatures = [...allSignatures, touchPoints];
      setAllSignatures(updatedAllSignatures);
      if (currentSignature === 10) {
        setAllSignatures([]);//////////////////
        setTouchPoints([]);
        setViewPoints([]);
        setCurrentPage('loadingPage_register');
        console.log(updatedAllSignatures.length)
  
        // Send all signatures to the server when 10 signatures are saved
        try {
          const response = await fetch(NGROK_LINK + '/TrainDots', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(updatedAllSignatures),
          });
          if (response.status === 200) {     
            /////////////////////
            try {
              const ganResponse = await fetch(NGROK_LINK + '/GAN', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                 body: JSON.stringify(updatedAllSignatures),
              });
              
              if (ganResponse.status !== 200) {
                throw new Error('Error with GAN request');
              }
              // Handle response data from GAN request if needed
              // const ganData = await ganResponse.json();
            } catch (ganError) {
              console.error(ganError);
              alert('Error with GAN request');
            }
            /////////////////////
            setCurrentPage('login');
          } else {
            alert('Error saving signatures');
          }
          setAllPaths([]);
        } catch (error) {
          console.error(error);
          alert('Error saving signatures');
        }
      } else {
        setCurrentSignature(currentSignature + 1);
        setTouchPoints([]); // Reset the touchPoints for the next signature
        setViewPoints([]);
        //clear drawing 
        setAllPaths([]);
      }
    }
  
    const handleClear = () => {
      setTouchPoints([]); // Reset the doodle after saving
      setViewPoints([]);
      setNotification('');
      // Clear the path
      setAllPaths([])
  
    };
  
    async function handleLogin() {
      setTouchPoints([]); // Reset the doodle after saving
      setViewPoints([]);
      setNotification('');
      setAllSignatures([]);
      setCurrentSignature(1);
      setCurrentPage('loadingPage_login');

  
      if (touchPoints.length <= 8) { //
        alert("Your singature is too short!")
        setTouchPoints([]); // Reset the doodle after saving
        setViewPoints([]);
        return;
      }
  
      const updatedAllSignatures = [touchPoints];
      setAllSignatures(updatedAllSignatures);
  
      try {
        const response = await fetch(NGROK_LINK + '/LoginDots', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(updatedAllSignatures),
        })
        .then((response) => response.text())
				.then((data) => {
					
					console.log(data);
          setCurrentPage('login');
          alert(data)
          setTouchPoints([]); // Reset the touchPoints for the next signature
          setViewPoints([]);
				})
				.catch((error) => {
					console.error('Error saving signatures:', error);
				});




       /* if (response.status === 200) {
          alert(response.text)
          //alert('The signature saved and sent to the server');
          setTouchPoints([]); // Reset the touchPoints for the next signature
          setViewPoints([]);
        } else {
          alert('Error saving signatures');
        }*/
        setAllPaths([])
  
      } catch (error) {
        console.error(error);
        alert('Error saving signatures');
      }
      setAllSignatures([]);
  
    };
  
    const handleMain = () => {
      setTouchPoints([]); // Reset the doodle after saving
      setViewPoints([]);
      setCurrentSignature(1);
      setCurrentPage('main'); // Update the currentPage state
      setAllPaths([])
      setAllSignatures([]); // Reset the allSignatures array
    };
  
    const renderPage = () => {
      if (currentPage === 'main') {
        return (
          <View style={styles.container}>
            <TouchableOpacity
              style={styles.button}
              onPress={() => setCurrentPage('login')}
            >
              <Text style={styles.buttonText}>Login</Text>
  
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.button}
              onPress={() => setCurrentPage('register')}
            >
              <Text style={styles.buttonText}>Register</Text>
            </TouchableOpacity>
          </View>
        );
      } else if (currentPage === 'login') {
        return (
          <View style={styles.container}{...panResponder.panHandlers}>
            <Text style={styles.modalTitle}>Login</Text>
  
          </View>
        );
      } else if (currentPage === 'register') {
        return (
          <View style={styles.container}{...panResponder.panHandlers}>
            <Text style={styles.modalTitle}>Register</Text>
            <Text >Sign below to learn the model:</Text>
            <Text >{`Signature ${currentSignature} out of 10`}</Text>
          </View>
        );
      } else if (currentPage === 'loadingPage_register') {
        return (
          <View style={styles.container}{...panResponder.panHandlers}>
            <ActivityIndicator
            color="#007AFF"
            size="large"
            />
              <Text style={styles.LoadingText}>Please wait for learning the model</Text>
              <Text style={styles.LoadingText}>Please do not close the application!</Text>
          </View>
        );
      }else if (currentPage === 'loadingPage_login') {
        return (
          <View style={styles.container}{...panResponder.panHandlers}>
            <ActivityIndicator
            color="#007AFF"
            size="large"
            />
              <Text style={styles.LoadingText}>Please wait while the signature is checked by the server..</Text>
          </View>
        );
      }
    };
    return (
      <View style={styles.container}>
        {renderPage()}
  
        {(currentPage === 'login') && (
  
          <View style={styles.container}>
  
            <View
              style={styles.signaturePadContainer}
              {...panResponder.panHandlers}
            >
              <View style={styles.signaturePad} >
                <Svg width={300} height={300}>
                  {allPaths.map((pathData, index) => (
                    <Path key={index} d={pathData} fill="none" stroke="#007AFF" strokeWidth={2} />
                  ))}
                  <Path d={currentPath} fill="none" stroke="#007AFF" strokeWidth={2} />
                </Svg>
              </View>
            </View>
            <View style={styles.buttonBlock}>
              <TouchableOpacity style={styles.buttonScreen} onPress={handleMain}>
                <Text style={styles.buttonText}>Back</Text>
              </TouchableOpacity><View><Text>          </Text></View>
              <TouchableOpacity style={styles.buttonScreen} onPress={handleLogin}>
                <Text style={styles.buttonText}>Login</Text>
              </TouchableOpacity><View><Text>          </Text></View>
              <TouchableOpacity style={styles.buttonScreen} onPress={handleClear}>
                <Text style={styles.buttonText}>Clear</Text>
              </TouchableOpacity>
  
              <Text>{notification}</Text>
            </View>
          </View>
        )}
  
        {(currentPage === 'register') && (
  
          <View style={styles.container} >
            <View
              style={styles.signaturePadContainer}
              {...panResponder.panHandlers}
            >
              <View style={styles.signaturePad} >
                <Svg width={300} height={300}>
                  {allPaths.map((pathData, index) => (
                    <Path key={index} d={pathData} fill="none" stroke="#007AFF" strokeWidth={2} />
                  ))}
                  <Path d={currentPath} fill="none" stroke="#007AFF" strokeWidth={2} />
                </Svg>
              </View>
            </View>
            <View style={styles.buttonBlock}>
              <TouchableOpacity style={styles.buttonScreen} onPress={handleMain}>
                <Text style={styles.buttonText}>Back</Text>
              </TouchableOpacity><View><Text>          </Text></View>
              {signaturesMaxedOut ? (
                <TouchableOpacity style={styles.buttonScreen} onPress={handleSave}>
                  <Text style={styles.buttonText}>Done</Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity style={styles.buttonScreen} onPress={handleSave}>
                  <Text style={styles.buttonText}>Save</Text>
                </TouchableOpacity>
              )}<View><Text>          </Text></View>
              <TouchableOpacity style={styles.buttonScreen} onPress={handleClear}>
                <Text style={styles.buttonText}>Clear</Text>
              </TouchableOpacity>
  
              <Text>{notification}</Text>
            </View>
          </View>
  
        )}
  
      </View>
    );
  };
  
  const styles = StyleSheet.create({
    container: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      marginTop: 30,
    },
    containerSign: {
      height: 20,
    },
    signaturePadContainer: {
      width: 300,
      height: 300,
      borderWidth: 1,
      top: -120,
    },
    signaturePad: {
      flex: 1,
    },
    button: {
      marginTop: 10,
      padding: 10,
      backgroundColor: '#007AFF',
      borderRadius: 10,
      marginVertical: 10,
      width: 200,
    },
    buttonText: {
      color: 'white',
      fontSize: 18,
      textAlign: 'center',
    },
    modalTitle: {
      fontSize: 24,
      fontWeight: 'bold',
      marginBottom: 20,
      color: '#007AFF',
    },
    buttonScreen: {
      marginTop: -20,
      padding: 10,
      backgroundColor: '#007AFF',
      borderRadius: 10,
      marginVertical: 10,
  
  
    },
    buttonBlock: {
      flexDirection: "row",
    },
    LoadingText: {
      fontWeight: 'bold',
      color: '#007AFF',
      marginBottom: 12,
    },
  
  });
  
  export default SignaturePad;
  