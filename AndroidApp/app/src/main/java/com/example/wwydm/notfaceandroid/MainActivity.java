package com.example.wwydm.notfaceandroid;

import android.Manifest;
import android.app.ActivityOptions;
import android.content.Intent;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.transition.Explode;
import android.view.View;
import android.view.Window;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        getWindow().requestFeature(Window.FEATURE_CONTENT_TRANSITIONS);
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 2);

        getWindow().setExitTransition(new Explode());
    }

    @Override
    public void onClick(View v) {
        startActivity(new Intent(MainActivity.this, faceDetectionActivity.class));
    }

    public void switchToCardView(View v) {
        startActivity(new Intent(MainActivity.this, photosOverview.class));
    }

    public void viewServer(View v) {
        startActivity(new Intent(MainActivity.this, faceDetectionServer.class));
//        getWindow().setEnterTransition(new Explode());
    }
}
