
package org.tensorflow.lite.examples.classification;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.media.MediaPlayer;
import android.media.ToneGenerator;
import android.os.Build;
import android.os.SystemClock;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.util.Size;
import android.util.TypedValue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

import org.tensorflow.lite.examples.classification.env.BorderedText;
import org.tensorflow.lite.examples.classification.env.ImageUtils;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Recognition;
import org.tensorflow.lite.examples.classification.tflite.Yolov8;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();
  private static final Size DESIRED_PREVIEW_SIZE = new Size(1920, 1080);
  private static final float TEXT_SIZE_DIP = 10;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private long lastProcessingTimeMs;
  private Integer sensorOrientation;

  private BorderedText borderedText;

  private static final int INPUT_SIZE = 224;
  protected static final int BATCH_SIZE = 1;
  protected static final int PIXEL_SIZE = 3;
  private static final boolean SAVE_PREVIEW_BITMAP = false;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;
  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment;
  }

  protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
    ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
    byteBuffer.order(ByteOrder.nativeOrder());
    int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    int pixel = 0;
    for (int i = 0; i < INPUT_SIZE; ++i) {
      for (int j = 0; j < INPUT_SIZE; ++j) {
        final int val = intValues[pixel++];
        byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
        byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
        byteBuffer.putFloat((val & 0xFF) / 255.0f);
      }
    }
    return byteBuffer;
  }
  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    int cropSize = INPUT_SIZE;

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);
    frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    cropSize, cropSize,
                    sensorOrientation, false);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

  }

  @Override
  protected void processImage() {
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    //final int cropSize = Math.min(previewWidth, previewHeight);

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @SuppressLint("MissingPermission")
          @Override
          public void run() {

            ByteBuffer byteBuffer = convertBitmapToByteBuffer(croppedBitmap);
            List<Recognition> results;

            final long startTime = SystemClock.uptimeMillis();
            try {
              results = detector.detect_task(
                      byteBuffer,
                      previewHeight,
                      previewWidth,
                      0.5f,
                      0.5f,
                      0.9f
              );
            } catch (Exception e) {
              throw new RuntimeException(e);
            }
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            System.out.println("Start time: " + startTime + " last processing time ms: " + lastProcessingTimeMs);

            LOGGER.v("Detect: %s", results);
            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showResults(results);
                  }}
                );
            readyForNextImage();
        }}
    );
  }
}
