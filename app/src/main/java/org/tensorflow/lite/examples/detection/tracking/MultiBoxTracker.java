/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tracking;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.text.TextUtils;
import android.util.Pair;
import android.util.TypedValue;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier.Recognition;

/** A tracker that handles non-max suppression and matches existing objects to new detections. */
public class MultiBoxTracker {
  private static final float TEXT_SIZE_DIP = 18;
  private static final float MIN_SIZE = 16.0f;
  private static final int[] COLORS = {
          Color.GREEN,
          Color.RED,
          Color.BLUE,
          Color.YELLOW,
          Color.CYAN,
          Color.MAGENTA,
          Color.WHITE,
          Color.parseColor("#55FF55"),
          Color.parseColor("#FFA500"),
          Color.parseColor("#FF8888"),
          Color.parseColor("#AAAAFF"),
          Color.parseColor("#FFFFAA"),
          Color.parseColor("#55AAAA"),
          Color.parseColor("#AA33AA"),
          Color.parseColor("#0D0068")
  };
  final List<Pair<Float, RectF>> screenRects = new LinkedList<Pair<Float, RectF>>();
  private final Logger logger = new Logger();
  private final Queue<Integer> availableColors = new LinkedList<Integer>();
  private final Paint boxPaint = new Paint();
  private final Paint paint = new Paint();
  private final float textSizePx;
  private final BorderedText borderedText;
  private Matrix frameToCanvasMatrix;
  private int frameWidth;
  private int frameHeight;
  private int sensorOrientation;

  private List<RectF> face_hands_medbox_rect = new ArrayList<>();
  private List<RectF> tablet_mouth_rect = new ArrayList<>();
  private List<RectF> tablet_hand_rect = new ArrayList<>();


  public MultiBoxTracker(final Context context) {
    for (final int color : COLORS) {
      availableColors.add(color);
    }

    boxPaint.setColor(Color.RED);
    boxPaint.setStyle(Style.STROKE);
    boxPaint.setStrokeWidth(10.0f);
    boxPaint.setStrokeCap(Cap.ROUND);
    boxPaint.setStrokeJoin(Join.ROUND);
    boxPaint.setStrokeMiter(100);

    textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
  }

  public synchronized void setFrameConfiguration(
          final int width, final int height, final int sensorOrientation) {
    frameWidth = width;
    frameHeight = height;
    this.sensorOrientation = sensorOrientation;
  }

  public synchronized void drawDebug(final Canvas canvas) {
    final Paint textPaint = new Paint();
    textPaint.setColor(Color.WHITE);
    textPaint.setTextSize(60.0f);

    final Paint boxPaint = new Paint();
    boxPaint.setColor(Color.RED);
    boxPaint.setAlpha(200);
    boxPaint.setStyle(Style.STROKE);

    for (final Pair<Float, RectF> detection : screenRects) {
      final RectF rect = detection.second;
      canvas.drawRect(rect, boxPaint);
      canvas.drawText("" + detection.first, rect.left, rect.top, textPaint);
      borderedText.drawText(canvas, rect.centerX(), rect.centerY(), "" + detection.first);
    }
  }

  public synchronized void trackResults(final List<Recognition> results, final List<Recognition> results_mouth, final List<Recognition> results_hand, final long timestamp) {
    logger.i("Processing %d results from %d", results.size(), timestamp);
    processResults(results, results_mouth, results_hand);
  }

  private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
  }

  public synchronized void draw(final Canvas canvas, RectF current_mouth_pos_on_screen) {
    final boolean rotated = sensorOrientation % 180 == -90;
    final float multiplier =
            Math.min(
                    canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
                    canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
    frameToCanvasMatrix =
            ImageUtils.getTransformationMatrix(
                    frameWidth,
                    frameHeight,
                    (int) (multiplier * (rotated ? frameHeight : frameWidth)),
                    (int) (multiplier * (rotated ? frameWidth : frameHeight)),
                    sensorOrientation,
                    false);

    draw_face_rect(canvas, face_hands_medbox_rect);
    if (current_mouth_pos_on_screen != null){
      current_mouth_pos_on_screen = single_mouth_rect(canvas, current_mouth_pos_on_screen);
      draw_detected_parts(tablet_mouth_rect, canvas, current_mouth_pos_on_screen, "mouth_pill");
    }

    final RectF hand = new RectF(60, 370, 180, 470);
    RectF hand2 = new RectF(60, 370, 180, 470);
   // drawsub_areas(canvas, hand);
//    hand2 = current_mouth_pos_on_screen;

    draw_detected_parts(tablet_hand_rect, canvas, hand2, "hand_pill");

    final RectF medbox = new RectF(60, 50, 210, 350);
    final RectF medbox2 = new RectF(60, 50, 210, 350);
   // drawsub_areas(canvas, medbox);
//    draw_detected_parts(tablet_hand_rect, canvas, medbox2, "medbox");
  }

  private void draw_face_rect(Canvas canvas, List<RectF> rects) {
    if (rects != null) {
      for (final RectF rect : rects) {
        float y_min = (300 - rect.top) * (640 / 300f);
        float x_min = rect.left * (480 / 300f);
        float y_max = (300 - rect.bottom) * (640 / 300f);
        float x_max = rect.right * (480 / 300f);
        RectF trackedPosface = new RectF(y_min, x_min, y_max, x_max);//y_min, x_min, y_max, x_max  右上角为原点
        getFrameToCanvasMatrix().mapRect(trackedPosface);
//        trackedPosface = boundary_check(trackedPosface);
//        System.out.println("********" + rect);
        boxPaint.setColor(Color.BLUE);
        boxPaint.setStrokeWidth(8.0f);
        canvas.drawRoundRect(trackedPosface, 4, 4, boxPaint);
//        final String labelString = !TextUtils.isEmpty(recognition.title) ? String.format("%s %.2f", recognition.title, (100 * recognition.detectionConfidence)) : String.format("%.2f", (100 * recognition.detectionConfidence));
//        borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top, "1" + "%", boxPaint);
      }
    }
  }

  private void drawsub_areas(Canvas canvas, RectF rect) {
    boxPaint.setColor(Color.RED);
    boxPaint.setStyle(Style.STROKE);
    boxPaint.setStrokeWidth(20.0f);
    getFrameToCanvasMatrix().mapRect(rect);
//    rect = boundary_check(rect);
    float cornerSize = Math.min(rect.width(), rect.height()) / 8.0f;
    canvas.drawRoundRect(rect, cornerSize, cornerSize, boxPaint);
  }

  private void draw_detected_parts(List<RectF> rects, Canvas canvas, RectF rect_screen, String mode) {
    if (rects != null) {
      float y_min = 0f;
      float y_max = 0f;
      float x_min = 0f;
      float x_max = 0f;
      for (final RectF rect : rects) {
        if(mode.equals("hand_pill") || mode.equals("medbox")) {
          y_min = (300 - rect.top) * ((rect_screen.right - rect_screen.left) / 300f) + rect_screen.left;
          y_max = (300 - rect.bottom) * ((rect_screen.right - rect_screen.left) / 300f) + rect_screen.left;
          x_min = rect.left * ((rect_screen.bottom - rect_screen.top) / 300f) + rect_screen.top;
          x_max = rect.right * ((rect_screen.bottom - rect_screen.top) / 300f) + rect_screen.top;
        }
        if(mode.equals("mouth_pill")){
          y_min = (rect.top) * ((rect_screen.right - rect_screen.left) / 300f) + rect_screen.left;
          y_max = (rect.bottom) * ((rect_screen.right - rect_screen.left) / 300f) + rect_screen.left;
          x_min = (300 - rect.left) * ((rect_screen.bottom - rect_screen.top) / 300f) + rect_screen.top;
          x_max = (300 - rect.right) * ((rect_screen.bottom - rect_screen.top) / 300f) + rect_screen.top;
        }

//        System.out.println("********" + (rect_screen.right - rect_screen.left) / 300f + "********" + 150 / 300f);
        RectF trackedPosPH = new RectF(y_min, x_min, y_max, x_max);//y_min, x_min, y_max, x_max  右上角为原点
        getFrameToCanvasMatrix().mapRect(trackedPosPH);
//        trackedPosPH = boundary_check(trackedPosPH);
        boxPaint.setColor(Color.RED);
        boxPaint.setStrokeWidth(8.0f);
        canvas.drawRoundRect(trackedPosPH, 4, 4, boxPaint);
//        final String labelString = !TextUtils.isEmpty(recognition.title) ? String.format("%s %.2f", recognition.title, (100 * recognition.detectionConfidence)) : String.format("%.2f", (100 * recognition.detectionConfidence));
//        borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top, "1" + "%", boxPaint);
      }
    }
  }
  
  private RectF single_mouth_rect(Canvas canvas, RectF rect) {
    float x_min = rect.left * (480 / 300f);
    float y_min =  (300 - rect.top) * (640 / 300f);
    float x_max =  rect.right * (480 / 300f);
    float y_max =  (300 - rect.bottom) * (640 / 300f);
    RectF trackedPosface = new RectF(y_min, x_min, y_max, x_max);//y_min, x_min, y_max, x_max  右上角为原点
//    System.out.println("********########" + trackedPosface);
    getFrameToCanvasMatrix().mapRect(trackedPosface);
//    System.out.println("********" + trackedPosface);
//    trackedPosface = boundary_check(trackedPosface);
    System.out.println("********" + trackedPosface);
//    boxPaint.setColor(Color.GREEN);
//    boxPaint.setStrokeWidth(8.0f);
//    canvas.drawRoundRect(trackedPosface, 4, 4, boxPaint);
    return new RectF(y_min, x_min, y_max, x_max);
  }

  private RectF boundary_check(RectF rect){
    float x_min = Math.max(0f, rect.left);
    float y_min = Math.max(0f, rect.top);
    float x_max = Math.min(479f, rect.right);
    float y_max = Math.min(639f, rect.bottom);
    return new RectF(x_min, y_min, x_max, y_max);
  }

  private void processResults(final List<Recognition> results, final List<Recognition> results_mouth, final List<Recognition> results_hand) {
    face_hands_medbox_rect.clear();
    tablet_mouth_rect.clear();
    tablet_hand_rect.clear();
    screenRects.clear();

    for (final Recognition result : results) {
//        System.out.println("********" + result.getLocation());
        face_hands_medbox_rect.add(result.getLocation());
    }

    for (final Recognition result : results_mouth) {
//        System.out.println("********" + result.getLocation());
      tablet_mouth_rect.add(result.getLocation());
    }

    for (final Recognition result : results_hand) {
//        System.out.println("********" + result.getLocation());
      tablet_hand_rect.add(result.getLocation());
    }

  }
}
