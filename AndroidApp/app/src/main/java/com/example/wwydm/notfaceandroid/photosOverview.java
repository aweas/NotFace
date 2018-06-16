package com.example.wwydm.notfaceandroid;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.media.Image;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.design.widget.BottomNavigationView;
import android.support.v4.app.ActivityCompat;
import android.support.v4.app.ActivityOptionsCompat;
import android.support.v4.util.Pair;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.view.MenuItem;
import android.view.View;
import android.widget.Adapter;
import android.widget.BaseAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;

public class photosOverview extends AppCompatActivity {

    private TextView mTextMessage;

    private BottomNavigationView.OnNavigationItemSelectedListener mOnNavigationItemSelectedListener
            = new BottomNavigationView.OnNavigationItemSelectedListener() {

        @Override
        public boolean onNavigationItemSelected(@NonNull MenuItem item) {
//            switch (item.getItemId()) {
//                case R.id.navigation_home:
//                    mTextMessage.setText(R.string.title_home);
//                    return true;
//                case R.id.navigation_dashboard:
//                    mTextMessage.setText(R.string.title_dashboard);
//                    return true;
//                case R.id.navigation_notifications:
//                    mTextMessage.setText(R.string.title_notifications);
//                    return true;
//            }
            return false;
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_photos_overview);

        BottomNavigationView navigation = (BottomNavigationView) findViewById(R.id.navigation);
        navigation.setOnNavigationItemSelectedListener(mOnNavigationItemSelectedListener);

        RecyclerView.Adapter<ImageOverviewAdapter.cardedImageHolder> imageAdapter = new ImageOverviewAdapter(this);

        RecyclerView rv_images = (RecyclerView) findViewById(R.id.rv_images);
        rv_images.setAdapter(imageAdapter);
        rv_images.setLayoutManager(new LinearLayoutManager(this));
    }

    public void showDetails(View v) {
        BitmapDrawable bmp_drawable = (BitmapDrawable)((ImageView)v).getDrawable();
        Bitmap bmp = bmp_drawable.getBitmap();
//        String filename = compressBitmap(bmp);

        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, 100, bytes);

        Intent intent = new Intent(photosOverview.this, photosDetails.class);
        intent.putExtra("bitmap", bytes.toByteArray());

        Pair<View, String> pair1 = Pair.create((View)findViewById(R.id.ll_header), "image_header");
        Pair<View, String> pair2 = Pair.create((View)findViewById(R.id.img), "image_details");
        Pair<View, String> pair3 = Pair.create((View)findViewById(R.id.tv_description), "image_desc");

        ActivityOptionsCompat options = ActivityOptionsCompat.makeSceneTransitionAnimation(photosOverview.this, pair1, pair2, pair3);

        startActivity(intent, options.toBundle());
    }

    private String compressBitmap(Bitmap bitmap) {
        String fileName = "myImage";//no .png or .jpg needed
        try {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, bytes);
            FileOutputStream fo = openFileOutput(fileName, Context.MODE_PRIVATE);
            fo.write(bytes.toByteArray());
            // remember close file output
            fo.close();
        } catch (Exception e) {
            e.printStackTrace();
            fileName = null;
        }
        return fileName;
    }
}
