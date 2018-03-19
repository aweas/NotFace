package com.example.wwydm.notfaceandroid;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    @Override
    public void onClick(View v) {
        startActivity(new Intent(MainActivity.this, faceDetectionActivity.class));
    }

    public void viewServer(View v) {
        startActivity(new Intent(MainActivity.this, faceDetectionServer.class));
    }
}
