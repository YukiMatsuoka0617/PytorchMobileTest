package com.example.pytorchtest;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    Button buttonCamera;
    Button buttonRecognition;
    ImageView imageView;
    TextView textView;
    TextView textView2;

    Bitmap bitmap = null;
    Module module = null;

    int REQUEST_CAPTURE_IMAGE = 100;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        findView();
        button();

        try {
            bitmap = BitmapFactory.decodeStream(getAssets().open("image.jpg"));
            module = Module.load(assetFilePath(this, "resnet.pt"));
            } catch (IOException e) {
                Log.e("PytorchHelloWorld", "Error reading assets", e);
                finish();
            }
        // showing image on UI
        imageView.setImageBitmap(bitmap);

        recognition(bitmap);
    }

    void findView(){
        buttonCamera = findViewById(R.id.buttonCamera);
        buttonRecognition = findViewById(R.id.buttonRecognition);
        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.result1Class);
        textView2 = findViewById(R.id.result1Score);
    }

    void button(){
        buttonCamera.setOnClickListener(this);
        buttonRecognition.setOnClickListener(this);
    }

    void recognition(Bitmap image){
        // preparing input tensor
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(image,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

        // running the model
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        // getting tensor content as java array of floats
        final float[] scores = outputTensor.getDataAsFloatArray();

        // searching for the index with maximum score
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }

        String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

        // showing className on UI
        textView.setText("Class:" + className);
        textView2.setText("Score:" + String.valueOf(maxScore));

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (REQUEST_CAPTURE_IMAGE == requestCode && resultCode == Activity.RESULT_OK) {
            Bitmap capturedImage = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(capturedImage);
            recognition(capturedImage);
        }
    }



    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.buttonCamera:
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, REQUEST_CAPTURE_IMAGE);
                break;

            case R.id.buttonRecognition:
                recognition(bitmap);
                break;
        }
    }

}