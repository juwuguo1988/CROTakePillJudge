/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class MainActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;//true;
  private static final String TF_OD_API_MODEL_FILE = "tflite_graph.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.8f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = true;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private Bitmap pillfaceBitmap;
  private Bitmap pillHandBitmap;
  private Bitmap pillMouthBitmap;
//  private Bitmap Medbox;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  private int detectedface = 0;
  private int facethreshold = 10;
  private boolean showdetecttabletmsg = true;//this varaible is to control whether to show the tablet-detect hint, only show it for once after 10 faces detected
  private boolean detectedpillhand = false;
  private boolean detectedpillmouth = false;
  private int cnt_mouth = 0;

  RectF screen_pos = new RectF();

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);

      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = -90;//rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);
    pillMouthBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);


    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);
    cropToFrameTransform = new Matrix();
//    cropToFrameTransform = frameToCropTransform;
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
//    trackingOverlay.addCallback(
//        new DrawCallback() {
//          @Override
//          public void drawCallback(final Canvas canvas) {
//            tracker.draw(canvas, screen_pos);
//            screen_pos = null;
//            if (isDebug()) {
//              tracker.drawDebug(canvas);
//            }
//          }
//        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @Override
  protected int processImage(int obj_to_detect) {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return -1;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    if (detectedface > facethreshold - 1) {
      pillHandBitmap = Bitmap.createBitmap(rgbFrameBitmap, 60, 10, 120, 100, null, false);
      pillHandBitmap = adjustPhotoRotation(pillHandBitmap, sensorOrientation);
    }
//    Medbox = Bitmap.createBitmap(rgbFrameBitmap, 60, 130, 150, 300, null, false);
//    Medbox = adjustPhotoRotation(Medbox, sensorOrientation);
//    int a = rgbFrameBitmap.getPixel(100, 100);
//    int b = submaprgbFrameBitmap.getPixel(100, 100);


    readyForNextImage();

//    final Canvas canvas = new Canvas(croppedBitmap);
//    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

//    final Canvas canvasPH = new Canvas(pillHandBitmap);
//    canvasPH.drawBitmap(submaprgbFrameBitmap, frameToCropTransformPH, null);
//    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP && detectedface > facethreshold) {
//      ImageUtils.saveBitmap(pillHandBitmap);
      int a = 0;
    }
    if (detectedface > facethreshold - 1) {
      int width = pillHandBitmap.getWidth();
      int height = pillHandBitmap.getHeight();
      Matrix matrixPH = new Matrix();
      matrixPH.preScale(300 / 100f, 300/120f);
      pillHandBitmap = Bitmap.createBitmap(pillHandBitmap, 0, 0, width, height, matrixPH, false);
//      System.out.println("********" + pillHandBitmap.getHeight() + "********" + pillHandBitmap.getWidth());
    }

//    int width = Medbox.getWidth();
//    int height = Medbox.getHeight();
//    Matrix matrixPH = new Matrix();
//    matrixPH.preScale(300 / 300f, 300 / 150f);
//    Medbox = Bitmap.createBitmap(Medbox, 0, 0, width, height, matrixPH, false);
//    System.out.println("********" + Medbox.getHeight() + "********" + Medbox.getWidth());

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);

//            final List<Classifier.Recognition> results_medbox = detector.recognizeImage(croppedBitmap);
//            System.out.println("********" + resultsPH);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();
            final List<Classifier.Recognition> mappedRecognitions_handpill =
                    new LinkedList<Classifier.Recognition>();
            final List<Classifier.Recognition> mappedRecognitions_mouthpill =
                    new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
//              final RectF location2 = result.getLocation();
              boolean hasface = false;
              if (location != null && result.getConfidence() >= minimumConfidence && result.getTitle().equals("face")) {
                hasface = true;
//                cropToFrameTransform.mapRect(location);
                get_face_img(location);
                result.setLocation(location);
                mappedRecognitions.add(result);
              }

              else if (location != null && result.getConfidence() >= minimumConfidence && result.getTitle().equals("medbox")) {
//                cropToFrameTransform.mapRect(location);
                result.setLocation(location);
                mappedRecognitions.add(result);
              }

              else if (location != null && result.getConfidence() >= minimumConfidence && result.getTitle().equals("hand")) {
//                cropToFrameTransform.mapRect(location);
                result.setLocation(location);
                mappedRecognitions.add(result);
              }

              else if (location != null && result.getConfidence() >= minimumConfidence && result.getTitle().equals("mouth")) {
//                cropToFrameTransform.mapRect(location);
                result.setLocation(location);
                mappedRecognitions.add(result);
                screen_pos = result.getLocation();
//                screen_pos = new RectF(location2.left, location2.top, location2.right, location2.bottom);
                get_mouth_img(location);
//                System.out.println("********########" + pillMouthBitmap.getWidth());
                final List<Classifier.Recognition> results_mouth = detector.recognizeImage(pillMouthBitmap);
                for (final Classifier.Recognition mouth_pill_result : results_mouth){
                  final RectF location_mouth_pill = mouth_pill_result.getLocation();
                  System.out.println("********########" + mouth_pill_result + "********########" + location_mouth_pill + "********########" + mouth_pill_result.getConfidence() + "********########" + mouth_pill_result);
                  if (location_mouth_pill != null && mouth_pill_result.getConfidence() >= minimumConfidence && mouth_pill_result.getTitle().equals("tablet")){
//                    System.out.println("********########" + pillMouthBitmap.getWidth());
                    mouth_pill_result.setLocation(location_mouth_pill);
                    mappedRecognitions_mouthpill.add(mouth_pill_result);
                  }
                }
              }

              if (hasface && detectedface < facethreshold + 1) {
                detectedface += 1;
              }
            }

            if (detectedface > facethreshold) {
              if (showdetecttabletmsg) {
                Toast toast = Toast.makeText(getApplicationContext(), "face detected, now detect tablet, please place the pill in the red box", Toast.LENGTH_SHORT);
                toast.show();
                showdetecttabletmsg = false;
              }
              final List<Classifier.Recognition> resultsPH = detector.recognizeImage(pillHandBitmap);
              for (final Classifier.Recognition resultPH : resultsPH) {
                RectF locationPH = resultPH.getLocation();
//                String label = resultPH.getTitle();
                if (locationPH != null && resultPH.getConfidence() >= minimumConfidence && resultPH.getTitle().equals("tablet")) {
//                  System.out.println("********########" + locationPH);
                  resultPH.setLocation(locationPH);
                  mappedRecognitions_handpill.add(resultPH);
                }
              }
            }
            tracker.trackResults(mappedRecognitions, mappedRecognitions_mouthpill, mappedRecognitions_handpill, currTimestamp);
            trackingOverlay.postInvalidate();
            computingDetection = false;
            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
//                    showFrameInfo(previewWidth + "x" + previewHeight);
//                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
//                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });
    return  -1;
  }

  private void get_face_img(RectF location){
    int y = (int)((300f - location.left) * (480 / 300f));
    int x = (int)((300f - location.bottom) * (640 / 300f));
    int width = (int)((location.bottom - location.top) * (640 / 300f));
    int height = (int)((location.left - location.right) * (480 / 300f));
    pillfaceBitmap = Bitmap.createBitmap(rgbFrameBitmap, x, y, width, height, null, false);
    pillfaceBitmap = adjustPhotoRotation(pillfaceBitmap, sensorOrientation);
//    ImageUtils.saveBitmap(pillMouthBitmap);
    int mouth_width = pillfaceBitmap.getWidth();
    int mouth_height = pillfaceBitmap.getHeight();
    Matrix matrixPH = new Matrix();
    matrixPH.preScale(300f / mouth_width, 300f / mouth_height);
    pillfaceBitmap = Bitmap.createBitmap(pillfaceBitmap, 0, 0, mouth_width, mouth_height, matrixPH, false);
    ImageUtils.saveBitmap(pillfaceBitmap);
//    System.out.println("********" + pillMouthBitmap.getHeight() + "********" + pillMouthBitmap.getWidth());
//    screen_pos = new RectF(x, 480f - y - height, x + width, 480f - y);
  }

  private void get_mouth_img(RectF location){
    int y = (int)((300f - location.left) * (480 / 300f));
    int x = (int)((300f - location.bottom) * (640 / 300f));
    int width = (int)((location.bottom - location.top) * (640 / 300f));
    int height = (int)((location.left - location.right) * (480 / 300f));
    pillMouthBitmap = Bitmap.createBitmap(rgbFrameBitmap, x, y, width, height, null, false);
    pillMouthBitmap = adjustPhotoRotation(pillMouthBitmap, sensorOrientation);
//    ImageUtils.saveBitmap(pillMouthBitmap);
    int mouth_width = pillMouthBitmap.getWidth();
    int mouth_height = pillMouthBitmap.getHeight();
    Matrix matrixPH = new Matrix();
    matrixPH.preScale(300f / mouth_width, 300f / mouth_height);
    pillMouthBitmap = Bitmap.createBitmap(pillMouthBitmap, 0, 0, mouth_width, mouth_height, matrixPH, false);
//    ImageUtils.saveBitmap(pillMouthBitmap);
    cnt_mouth += 1;
//    System.out.println("********" + pillMouthBitmap.getHeight() + "********" + pillMouthBitmap.getWidth());
//    screen_pos = new RectF(x, 480f - y - height, x + width, 480f - y);
  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }

  public static Bitmap adjustPhotoRotation(Bitmap bitmap, int degree){
    if(bitmap!=null){
      Matrix m=new Matrix();
      try{
        m.setRotate(degree, bitmap.getWidth()/2, bitmap.getHeight()/2);//90就是我们需要选择的90度
        Bitmap bmp2=Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), m, true);
        bitmap.recycle();
        bitmap=bmp2;
      }catch(Exception ex){
        System.out.print("创建图片失败！"+ex);
      }
    }
    return bitmap;
  }
}
