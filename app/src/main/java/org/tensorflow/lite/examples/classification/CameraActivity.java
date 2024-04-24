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

package org.tensorflow.lite.examples.classification;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Fragment;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.AudioManager;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.media.MediaPlayer;
import android.media.ToneGenerator;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.os.Trace;
import android.os.VibrationEffect;
import android.os.Vibrator;
import androidx.annotation.NonNull;
import androidx.annotation.UiThread;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import com.google.android.material.bottomsheet.BottomSheetBehavior;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Objects;

import org.tensorflow.lite.examples.classification.env.ImageUtils;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Recognition;
import org.tensorflow.lite.examples.classification.tflite.Yolov8;

public abstract class CameraActivity extends AppCompatActivity implements OnImageAvailableListener,
                                                                          Camera.PreviewCallback,
                                                                          //View.OnClickListener,
                                                                          AdapterView.OnItemSelectedListener {
  private static final Logger LOGGER = new Logger();
  private static final int PERMISSIONS_REQUEST = 1;
  private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
  protected int previewWidth = 0;
  protected int previewHeight = 0;
  private Handler handler;
  private HandlerThread handlerThread;
  private boolean useCamera2API;
  private boolean isProcessingFrame = false;
  private byte[][] yuvBytes = new byte[3][];
  private int[] rgbBytes = null;
  private int yRowStride;
  private Runnable postInferenceCallback;
  private Runnable imageConverter;
  protected TextView textView;
  protected Button settings;
  private CameraManager manager;
  private int numThreads = -1;
  protected Yolov8 detector;
  private ToneGenerator toneGen1;
  private MediaPlayer five_hundred, one_thousand, two_thousand;
  private Vibrator vibrate;
  protected static final String MODEL_PATH = "best_float16.tflite";
  protected static final String LABELS_PATH = "labels.txt";

  @SuppressLint("MissingInflatedId")
  @Override
  protected void onCreate(final Bundle savedInstanceState) {

    try {
      detector = new Yolov8(
              getApplicationContext(),
              MODEL_PATH,
              true,
              3,
              true,
              LABELS_PATH,
              90
      );
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    LOGGER.d("onCreate " + this);
    super.onCreate(null);

    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    setContentView(R.layout.activity_camera);
    Toolbar toolbar = findViewById(R.id.toolbar);
    setSupportActionBar(toolbar);
    Objects.requireNonNull(getSupportActionBar()).setDisplayShowTitleEnabled(false);

    if (hasPermission()) { // checks if the user has permission to the camera whether it is granted or not
      setFragment();
    } else {
      requestPermission();
    }
    // settings
    settings = (Button) findViewById(R.id.settingsButton);
    settings.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        Intent i = new Intent(CameraActivity.this, SettingsActivity.class);
        startActivity(i);
      }
    });
    // beeps if the app is working
    toneGen1 = new ToneGenerator(AudioManager.STREAM_MUSIC, 400);
    toneGen1.startTone(ToneGenerator.TONE_PROP_BEEP2,200);

    // text for the currency recognition
    textView = (TextView) findViewById(R.id.simpleTextView);

    // initializing the vibration
    vibrate = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);

    // initializing mp3 voice files
    five_hundred = MediaPlayer.create(this, R.raw.fivehundred);
    one_thousand = MediaPlayer.create(this, R.raw.onethousand);
    two_thousand = MediaPlayer.create(this, R.raw.twothousand);
  }

  protected int[] getRgbBytes() {
    imageConverter.run();
    return rgbBytes;
  }

  /** Callback for android.hardware.Camera API */
  @Override
  public void onPreviewFrame(final byte[] bytes, final Camera camera) {

    // if we are processing a frame right now i want you to drop it and start processing frames after it's free
    if (isProcessingFrame) {
      LOGGER.w("Dropping frame!");
      return;
    }
    if(detector == null){
      LOGGER.w("Dropping frame! because model isn't initialized.");
      return;
    }

      camera.autoFocus(new Camera.AutoFocusCallback() {
        @Override
        public void onAutoFocus(boolean focused, Camera camera) {

        }
      });

//      camera.autoFocus(new Camera.AutoFocusCallback() {
//                         @Override
//                         public void onAutoFocus(boolean focused, Camera camera) {
//
//                           if (focused) {
                             isProcessingFrame = true; // we start processing the frames
                             //toneGen1.startTone(ToneGenerator.TONE_PROP_BEEP2, 150);
                             try {
                               // Initialize the storage bitmaps once when the resolution is known.
                               if (rgbBytes == null) {
                                 Camera.Size previewSize = camera.getParameters().getPreviewSize();
                                 previewHeight = previewSize.height;
                                 previewWidth = previewSize.width;
                                 rgbBytes = new int[previewWidth * previewHeight];
                                 onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
                               }

                             } catch (final Exception e) {
                               LOGGER.e(e, "Exception!");
                               return;
                             }
                             yuvBytes[0] = bytes; // raw bytes from camera in the yuv format

                             yRowStride = previewWidth; // number of the bytes in a pixel row

                             imageConverter =
                                     new Runnable() {
                                       @Override
                                       public void run() {
                                         ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);
                                       }
                                     }; // we are running this in the background

                             postInferenceCallback =
                                     new Runnable() {
                                       @Override
                                       public void run() {
                                         camera.addCallbackBuffer(bytes);
                                         isProcessingFrame = false;
                                       }
                                     };
                             processImage();

//                           }
//                         }
//
//                       }
//      );
    }


  /** Callback for Camera2 API */
  @Override
  public void onImageAvailable(final ImageReader reader) {
    // We need wait until we have some size from onPreviewSizeChosen
    if (previewWidth == 0 || previewHeight == 0) {
      return;
    }
    if (rgbBytes == null) {
      rgbBytes = new int[previewWidth * previewHeight];
    }
    try {
      final Image image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      if (isProcessingFrame) {
        image.close();
        return;
      }
      isProcessingFrame = true;
      Trace.beginSection("imageAvailable");
      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);
      yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();

      imageConverter =
          new Runnable() {
            @Override
            public void run() {
              ImageUtils.convertYUV420ToARGB8888(
                  yuvBytes[0],
                  yuvBytes[1],
                  yuvBytes[2],
                  previewWidth,
                  previewHeight,
                  yRowStride,
                  uvRowStride,
                  uvPixelStride,
                  rgbBytes);
            }
          };

      postInferenceCallback =
          new Runnable() {
            @Override
            public void run() {
              image.close();
              isProcessingFrame = false;
            }
          };

      processImage();
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }
    Trace.endSection();
  }

  @Override
  public synchronized void onStart() {
    LOGGER.d("onStart " + this);
    super.onStart();

  }

  @Override
  public synchronized void onResume() {
    LOGGER.d("onResume " + this);
    super.onResume();

    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
  }

  @Override
  public synchronized void onPause() {
    LOGGER.d("onPause " + this);

    handlerThread.quitSafely();
    try {
      handlerThread.join();
      handlerThread = null;
      handler = null;
    } catch (final InterruptedException e) {
      LOGGER.e(e, "Exception!");
    }

    super.onPause();
  }

  @Override
  public synchronized void onStop() {
    LOGGER.d("onStop " + this);
    super.onStop();
  }

  @Override
  public synchronized void onDestroy() {
    LOGGER.d("onDestroy " + this);
    super.onDestroy();
  }

  // this is a method that posts a mission to a thread that running in the background
  protected synchronized void runInBackground(final Runnable r) {
    if (handler != null) {

      handler.post(r);
    }
  }
  @Override
  public void onRequestPermissionsResult(
      final int requestCode, final String[] permissions, final int[] grantResults) {
    if (requestCode == PERMISSIONS_REQUEST) {
      if (allPermissionsGranted(grantResults)) {
        setFragment();
      } else {
        requestPermission();
      }
    }
  }
  private static boolean allPermissionsGranted(final int[] grantResults) {
    for (int result : grantResults) {
      if (result != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }
  private boolean hasPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
    } else {
      return true;
    }
  }
  private void requestPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
        Toast.makeText(
                CameraActivity.this,
                "Camera permission is required for this demo",
                Toast.LENGTH_LONG)
            .show();
      }
      requestPermissions(new String[] {PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
    }
  }
  private boolean isHardwareLevelSupported(CameraCharacteristics characteristics, int requiredLevel) {
    int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
    if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
      return requiredLevel == deviceLevel;
    }
    // deviceLevel is not LEGACY, can use numerical sort
    return requiredLevel <= deviceLevel;
  }
  private String chooseCamera() {
    this.manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
    try {

      for (final String cameraId : manager.getCameraIdList()) {
        final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

        // We don't use a front facing camera in this sample.
        final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
        if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
          continue;
        }

        final StreamConfigurationMap map =
            characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

        if (map == null) {
          continue;
        }

        // Fallback to camera1 API for internal cameras that don't have full support.
        // This should help with legacy situations where using the camera2 API causes
        // distorted or otherwise broken previews.
        useCamera2API =
            (facing == CameraCharacteristics.LENS_FACING_EXTERNAL)
                || isHardwareLevelSupported(
                    characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL);

        LOGGER.i("Camera API lv2?: %s", useCamera2API);
        return cameraId;
      }
    } catch (CameraAccessException e) {
      LOGGER.e(e, "Not allowed to access camera");
    }

    return null;
  }

  protected void setFragment() {
    String cameraId = chooseCamera();

    Fragment fragment;
    if (useCamera2API) { // false
      CameraConnectionFragment camera2Fragment =
          CameraConnectionFragment.newInstance(
              new CameraConnectionFragment.ConnectionCallback() {
                @Override
                public void onPreviewSizeChosen(final Size size, final int rotation) {
                  previewHeight = size.getHeight();
                  previewWidth = size.getWidth();
                  CameraActivity.this.onPreviewSizeChosen(size, rotation);
                }
              },
              this,
              getLayoutId(),
              getDesiredPreviewFrameSize()
                  );

      camera2Fragment.setCamera(cameraId);
      fragment = camera2Fragment;
    } else {

      fragment =
          new LegacyCameraConnectionFragment(
                  this, getLayoutId(), getDesiredPreviewFrameSize());
    }
    // here replace the container with a camera fragment
    getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
  }

  protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (int i = 0; i < planes.length; ++i) {
      final ByteBuffer buffer = planes[i].getBuffer();
      if (yuvBytes[i] == null) {
        LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
        yuvBytes[i] = new byte[buffer.capacity()];
      }
      buffer.get(yuvBytes[i]);
    }
  }
  protected void readyForNextImage() {
    if (postInferenceCallback != null) {
      postInferenceCallback.run();
    }
  }
  protected int getScreenOrientation() {
    switch (getWindowManager().getDefaultDisplay().getRotation()) {
      case Surface.ROTATION_270:
        return 270;
      case Surface.ROTATION_180:
        return 180;
      case Surface.ROTATION_90:
        return 90;
      default:
        return 0;
    }
  }
  @SuppressLint("MissingPermission")
  @UiThread
  protected void showResults(List<Recognition> results) {

    if (results != null && results.size() >= 1) {
      Recognition recognition = results.get(0);

      //if (recognition.getTitle().equals(recognition1.getTitle()) && recognition2.getTitle().equals(recognition1.getTitle())) {
          // produce tone to make sure that we detected a currency
          toneGen1.startTone(ToneGenerator.TONE_CDMA_CONFIRM,150);
          //
          if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrate.vibrate(VibrationEffect.createOneShot(150, VibrationEffect.DEFAULT_AMPLITUDE));
          } else {
            //deprecated in API 26
            vibrate.vibrate(new long[]{0, 500}, -1);
          }
          try {
            switch (recognition.getTitle()) {
              case "500":
                five_hundred.start();
                break;
              case "1000":
                one_thousand.start();
                break;
              case "2000":
                two_thousand.start();
                break;
            }
          }catch (Exception e){
            e.printStackTrace();
          }



        if (recognition.getTitle() != null){
          textView.setText(recognition.getTitle());
        }


    }else{
      textView.setText(" ");
    }
  }
  protected int getNumThreads() {
    return numThreads;
  }
  protected abstract void processImage();

  protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

  protected abstract int getLayoutId();

  protected abstract Size getDesiredPreviewFrameSize();




  @Override
  public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {

  }

  @Override
  public void onNothingSelected(AdapterView<?> parent) {
    // Do nothing.
  }
}
