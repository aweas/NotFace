package com.example.wwydm.notfaceandroid;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.ImageView;

public class photosDetails extends AppCompatActivity {
    ImageView main_photo;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Intent intent = getIntent();
        Bundle b = intent.getExtras();
//        String filename = (String)b.get("bitmap");
//        Bitmap bmp = decompressBitmap(filename);
        byte[] bytes = (byte[])b.get("bitmap");
        Bitmap bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);

        setContentView(R.layout.activity_photos_details);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        main_photo = findViewById(R.id.img);
        main_photo.setImageBitmap(bmp);
    }

    private Bitmap decompressBitmap(String filename) {
        Bitmap bmp = BitmapFactory.decodeFile(filename);
        return bmp;
    }
}
