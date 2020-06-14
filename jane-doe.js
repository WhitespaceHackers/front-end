/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// import * as posenet_module from "@tensorflow-models/posenet";
// import * as facemesh_module from "@tensorflow-models/facemesh";
// import * as tf from "@tensorflow/tfjs";
import * as paper from "paper";
import "babel-polyfill";

import dat from "dat.gui";
import { SVGUtils } from "./utils/svgUtils";
import { PoseIllustration } from "./illustrationGen/illustration";
import { Skeleton } from "./illustrationGen/skeleton";
import { toggleLoadingUI, setStatusText } from "./utils/demoUtils";

import * as boySVG from "./resources/illustration/boy.svg";
import * as girlSVG from "./resources/illustration/girl.svg";
import * as abstractSVG from "./resources/illustration/abstract.svg";
import * as blathersSVG from "./resources/illustration/blathers.svg";
import * as tomNookSVG from "./resources/illustration/tom-nook.svg";
import * as boy_doughnut from "./resources/images/boy_doughnut.jpg";
import * as tie_with_beer from "./resources/images/tie_with_beer.jpg";
import * as test_img from "./resources/images/test.png";
import * as full_body from "./resources/images/full-body.png";
import * as full_body_1 from "./resources/images/full-body_1.png";
import * as full_body_2 from "./resources/images/full-body_2.png";

// clang-format off
import {
  drawKeypoints,
  drawPoint,
  drawSkeleton,
  renderImageToCanvas,
} from "./utils/demoUtils";
import { FileUtils } from "./utils/fileUtils";

// clang-format on
const resnetArchitectureName = "MobileNetV1";
const avatarSvgs = {
  girl: girlSVG.default,
  boy: boySVG.default,
  abstract: abstractSVG.default,
  blathers: blathersSVG.default,
  "tom-nook": tomNookSVG.default,
};
const sourceImages = {
  boy_doughnut: boy_doughnut.default,
  tie_with_beer: tie_with_beer.default,
  test_img: test_img.default,
  full_body: full_body.default,
  full_body_1: full_body_1.default,
  full_body_2: full_body_2.default,
};

let skeleton;
let illustration;
let canvasScope;

let posenet;
let facemesh;

const VIDEO_WIDTH = 513;
const VIDEO_HEIGHT = 513;

const CANVAS_WIDTH = 513;
const CANVAS_HEIGHT = 513;

const defaultQuantBytes = 2;
const defaultMultiplier = 1.0;
const defaultStride = 16;
const defaultInputResolution = 257;
const defaultMaxDetections = 1;
const defaultMinPartConfidence = 0.5;
const defaultMinPoseConfidence = 0.5;
const defaultNmsRadius = 20.0;

let predictedPoses;
let faceDetection;
let sourceImage;

/**
 * Draws a pose if it passes a minimum confidence onto a canvas.
 * Only the pose's keypoints that pass a minPartConfidence are drawn.
 */
function drawResults(image, canvas, faceDetection, poses) {
  // renderImageToCanvas(image, [VIDEO_WIDTH, VIDEO_HEIGHT], canvas);
  const ctx = canvas.getContext("2d");
  poses.forEach((pose) => {
    if (pose.score >= defaultMinPoseConfidence) {
      // if (guiState.showKeypoints) {
      drawKeypoints(pose.keypoints, defaultMinPartConfidence, ctx);
      // }

      // if (guiState.showSkeleton) {
      drawSkeleton(pose.keypoints, defaultMinPartConfidence, ctx);
      // }
    }
  });
  // if (guiState.showKeypoints) {
  //   faceDetection.forEach(face => {
  //     Object.values(facePartName2Index).forEach(index => {
  //         let p = face.scaledMesh[index];
  //         drawPoint(ctx, p[1], p[0], 3, 'red');
  //     });
  //   });
  // }
}

async function loadImage(imagePath) {
  const image = new Image();
  const promise = new Promise((resolve, reject) => {
    image.crossOrigin = "";
    image.onload = () => {
      resolve(image);
    };
  });

  image.src = imagePath;
  return promise;
}

function multiPersonCanvas() {
  return document.querySelector("#multi canvas");
}

function getIllustrationCanvas() {
  return document.querySelector(".illustration-canvas");
}

/**
 * Draw the results from the multi-pose estimation on to a canvas
 */
function drawDetectionResults() {
  const canvas = multiPersonCanvas();
  drawResults(null, canvas, faceDetection, predictedPoses);
  if (!predictedPoses || !predictedPoses.length || !illustration) {
    return;
  }

  skeleton.reset();
  canvasScope.project.clear();

  illustration.updateSkeleton(predictedPoses[0], null);
  illustration.draw(canvasScope, 1000, 500);

  // if (guiState.showCurves) {
  //   illustration.debugDraw(canvasScope);
  // }
  // if (guiState.showLabels) {
  //   illustration.debugDrawLabel(canvasScope);
  // }
}

/**
 * Loads an image, feeds it into posenet the posenet model, and
 * calculates poses based on the model outputs
 */
async function testImageAndEstimatePoses() {
  toggleLoadingUI(true);
  // setStatusText("Loading FaceMesh model...");
  // document.getElementById("results").style.display = "none";

  // Reload facemesh model to purge states from previous runs.
  // facemesh = await facemesh_module.load();

  // Load an example image
  // setStatusText("Loading image...");
  // sourceImage = await loadImage(sourceImages[guiState.sourceImage]);

  // Estimates poses
  // setStatusText("Predicting...");
  // predictedPoses = await posenet.estimatePoses(sourceImage, {
  //   flipHorizontal: false,
  //   decodingMethod: 'multi-person',
  //   maxDetections: defaultMaxDetections,
  //   scoreThreshold: defaultMinPartConfidence,
  //   nmsRadius: defaultNmsRadius,
  // });

  console.log("awaiting data")
  let response = await fetch(
    "https://mdjj-api.us-south.cf.appdomain.cloud/api"
  );
console.log("data received")
  let data = await response.json();

  console.log(data.fall, data.heartRate, data.steps);

  document.getElementById("steps").innerHTML = data.steps;
  document.getElementById("heart-rate").innerHTML = data.heartRate;

  if (data.fall) {
    var today = new Date();
    var dd = today.getDate();
    var mm = today.getMonth() + 1;
    var yyyy = today.getFullYear();
    var h = today.getHours();
    var m = today.getMinutes();
    var s = today.getSeconds();
    if (dd < 10) {
      dd = "0" + dd;
    }

    if (mm < 10) {
      mm = "0" + mm;
    }

    if (m < 10) {
      m = "0" + m;
    }

    if (s < 10) {
      s = "0" + s;
    }

    var dateString = today = dd+'/'+mm+'/'+yyyy;
    var timeString = h + ":" + m + ":" + s;
    var fallMessage = "FALL DETECTED";
    var fallTime = dateString + " " + timeString; 
    
    document.getElementById("fall-status").innerHTML = fallMessage;
    document.getElementById("fall-time").innerHTML = fallTime;

  }

  predictedPoses = [data];
  console.log(predictedPoses);

  // faceDetection = await facemesh.estimateFaces(sourceImage, false, false);

  // Draw poses.
  drawDetectionResults();

  toggleLoadingUI(false);
  document.getElementById("results").style.display = "block";
}

let guiState = {
  // Selected image
  // sourceImage: Object.keys(sourceImages)[0],
  avatarSVG: Object.keys(avatarSvgs)[0],
  // Detection debug
  showKeypoints: true,
  showSkeleton: true,
  // Illustration debug
  showCurves: false,
  showLabels: false,
};

function setupGui() {
  const gui = new dat.GUI();

  const imageControls = gui.addFolder("Image");
  imageControls.open();
  gui
    .add(guiState, "sourceImage", Object.keys(sourceImages))
    .onChange(() => testImageAndEstimatePoses());
  gui
    .add(guiState, "avatarSVG", Object.keys(avatarSvgs))
    .onChange(() => loadSVG(avatarSvgs[guiState.avatarSVG]));

  const debugControls = gui.addFolder("Debug controls");
  debugControls.open();
  gui.add(guiState, "showKeypoints").onChange(drawDetectionResults);
  gui.add(guiState, "showSkeleton").onChange(drawDetectionResults);
  gui.add(guiState, "showCurves").onChange(drawDetectionResults);
  gui.add(guiState, "showLabels").onChange(drawDetectionResults);
}

/**
 * Kicks off the demo by loading the posenet model and estimating
 * poses on a default image
 */
export async function bindPage() {
  toggleLoadingUI(true);
  canvasScope = paper.default;
  let canvas = getIllustrationCanvas();
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;
  canvasScope.setup(canvas);

  // await tf.setBackend('webgl');
  // setStatusText("Loading PoseNet model...");
  // posenet = await posenet_module.load({
  //   architecture: resnetArchitectureName,
  //   outputStride: defaultStride,
  //   inputResolution: defaultInputResolution,
  //   multiplier: defaultMultiplier,
  //   quantBytes: defaultQuantBytes
  // });

  // setupGui(posenet);
  // setStatusText("Loading SVG file...");
  await loadSVG(Object.values(avatarSvgs)[0]);
}

window.onload = bindPage;
FileUtils.setDragDropHandler(loadSVG);

// Target is SVG string or path
async function loadSVG(target) {
  let svgScope = await SVGUtils.importSVG(target);
  skeleton = new Skeleton(svgScope);
  illustration = new PoseIllustration(canvasScope);
  illustration.bindSkeleton(skeleton, svgScope);
  setInterval(() => {
    console.log("refetching pose data");
    testImageAndEstimatePoses();
  }, 5000);
}
