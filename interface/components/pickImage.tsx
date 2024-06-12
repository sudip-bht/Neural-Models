import React ,{ useState } from "react";
import {
  View,
  Button,
  Alert,
  Image,
  Text,
  StyleSheet,
  ImageBackground,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { Picker } from "@react-native-picker/picker";
import * as FileSystem from "expo-file-system"
import pako from "pako"


import { NativeWindStyleSheet } from "nativewind";
import { red } from "react-native-reanimated/lib/typescript/reanimated2/Colors";
import { FullWindowOverlay } from "react-native-screens";






export default function PickImage() {
  const [image, setImage] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [prediction, setPrediction] = useState("");
  const [base64String, setBase64String] = useState("");

  let pickImageButtonLabel = image
    ? "cancel image"
    : "Pick an image from camera roll";

  const sendStringsToServer = async () => {
    try {
      console.log("whats going on");

      const response = await fetch(
        "http://172.18.161.171:5000//process_strings",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            string1: base64String,
            string2: selectedModel,
          }),
        }
      );
      console.log("above data");

      const data = await response.json();

      if (data) {
        setBase64String("");
        setPrediction(data["prediction"]);
      }
      console.log("below data");

      console.log(data);
    } catch (error) {
      console.error("Error sending strings to server:", error);
    }
  };

  const pickImage = async () => {
    if (!image) {
      // No permissions request is necessary for launching the image library
      let result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.All,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 1,
      });

      // if (result.assets) {
      //   console.log(result.assets[0].fileName);
      //   console.log(typeof(result.assets[0]));
      //   console.log(result.assets[0]);
      // }

      if (!result.canceled) {
        setImage(result.assets[0].uri);
      }
    } else {
      setImage("");
      setPrediction("");
    }
  };

 

  const convertToBase64 = async () => {
    try {
      const fileInfo = await FileSystem.getInfoAsync(image);
      if (fileInfo.exists) {
        const base64 = await FileSystem.readAsStringAsync(image, {
          encoding: FileSystem.EncodingType.Base64,
        });

       
        setBase64String(base64);
        console.log("after setting");
        console.log(base64);
      } else {
        console.error("File does not exist:", image);
      }
    } catch (error) {
      console.error("Error converting to base64:", error);
    }
  };

  return (
    <View style={styles.container}>
      <ImageBackground
        source={require("../assets/images/bg-skin.jpg")}
        style={styles.background}
      >
        <View style={styles.body}>
          <Text style={{ fontSize: 15, fontWeight:"bold" }}>Select the algorithm to use</Text>
          <Picker
            selectedValue={selectedModel}
            placeholder="Select the algorithm to use"
            onValueChange={(itemValue, itemIndex) =>
              setSelectedModel(itemValue)
            }
            dropdownIconColor={"blue"}
         
            style={styles.picker}
          >
            <Picker.Item label="ResNext" value="resnext" />
            <Picker.Item label="Inception v4" value="inceptionv4" />
            <Picker.Item label="Inception v3" value="inceptionv3" />
            <Picker.Item label="Yolo v8" value="yolov8" />
            <Picker.Item label="Cancel" value="" />
            <Picker.Item label="None" value="" />
          </Picker>

          {image && <Image source={{ uri: image }} style={styles.image} />}
          {prediction && <Text style={styles.text}>{prediction}</Text>}
          <View style={styles.buttonContainer}>
            <Button
              title={pickImageButtonLabel}
              onPress={pickImage}
              color={""}
            />
          </View>
          <View style={styles.buttonContainer}>
            <Button
              title="Find the result"
              onPress={() => {
                console.log("submit pressed");

                if (!image) {
                  Alert.alert("No Image", "Please upload the image first", [
                    {
                      text: "Cancel",
                      onPress: () => console.log("Cancel Pressed"),
                      style: "cancel",
                    },
                    { text: "OK", onPress: () => console.log("OK Pressed") },
                  ]);
                }
                if (!selectedModel) {
                  Alert.alert("No Model", "Please select the model", [
                    {
                      text: "Cancel",
                      onPress: () => console.log("Cancel Pressed"),
                      style: "cancel",
                    },
                    { text: "OK", onPress: () => console.log("OK Pressed") },
                  ]);
                }
                if (image && selectedModel) {
                  console.log("submit button pressed");
                  convertToBase64();
                  console.log(base64String);

                  if (base64String) {
                    console.log("base 64 obtained");
                    sendStringsToServer();
                  }
                }
              }}
            />
          </View>
        </View>
      </ImageBackground>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    height: "100%",
    width: "100%",

    backgroundColor: "lightgrey",
  },
  background: {
    height: "100%",
    width: "100%",
    resizeMode: "cover",
  },
  body: {
    flex: 1,
    justifyContent: "center",
    padding: 60,
  },
  picker: {
    marginBottom: 20,
    borderWidth: 10,
    
    fontSize: 10,
    
  },
  buttonContainer: {
    marginBottom: 20,
  },
  image: {
    width: 200,
    height: 200,
    alignSelf: "center",
    marginBottom: 20,
  },
  text: {
    textAlign: "center",
    marginBottom: 20,
  },
});