package com.example.wwydm.notfaceandroid;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;
import java.io.FilenameFilter;
import java.util.Arrays;

public class faceDetectionActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_face_detection);

        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
        fab.setVisibility(View.GONE);

        ImageView img_face = findViewById(R.id.imageView);
        img_face.setImageResource(R.drawable.face);

        ImageView img_face2 = findViewById(R.id.imageView2);
        img_face2.setImageResource(R.drawable.not_face);
    }

    public void writeFilePath(View v) {
        Bitmap bmp = ((BitmapDrawable)((ImageView)v).getDrawable()).getBitmap();
        Bitmap resized = Bitmap.createScaledBitmap(bmp, 64, 64, true);

        int[] rgb_map = getRGBMap(resized);

        ((ImageView)findViewById(R.id.iv_display)).setImageBitmap(resized);

        debugFiles();

//        recognizeFace(rgb_map);
    }

    private void debugFiles() {
        File file = new File("file:///android_asset/");
        String[] directories = file.list(new FilenameFilter() {
            @Override
            public boolean accept(File current, String name) {
                return new File(current, name).isDirectory();
            }
        });
        System.out.println(Arrays.toString(directories));
    }

    private void recognizeFace(int[] rgb_map) {
        float[] outputs = new float[2];

        TensorFlowInferenceInterface tensorflow = new TensorFlowInferenceInterface(getAssets(), "file:///android_asset/raw/model.pb");
        tensorflow.feed("main_input", rgb_map, 1, 64, 64, 3);
        tensorflow.run(new String[]{"main_output/Softmax"}, false);
        tensorflow.fetch("main_output/Softmax", outputs);

        ((TextView)findViewById(R.id.tv_bitmap)).setText(Float.toString(outputs[0]));
    }

    private int[] getRGBMap(Bitmap resized) {
        int[] pixels = new int[64*64];
        int [] rgb_map = new int[64*64*3];
        resized.getPixels(pixels, 0, 64, 0, 0, 64, 64);

        for(int i=0; i<64*64; i++){
            int R = Color.red(pixels[i]);
            int G = Color.green(pixels[i]);
            int B = Color.blue(pixels[i]);

            rgb_map[i*3] = R;
            rgb_map[i*3+1] = G;
            rgb_map[i*3+2] = B;
        }
        return rgb_map;
    }
}
