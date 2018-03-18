package com.example.wwydm.notfaceandroid;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.media.Image;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Base64;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.Console;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;

public class faceDetectionServer extends AppCompatActivity {
    ImageView iv_display;
    Bitmap currentBitmap;
    String result;
    String IP;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_face_detection_server);

        initializeGUI();
    }

    private void initializeGUI() {
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
//        fab.setImageResource(R.drawable.face);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, 1888);
            }
        });

        ImageView img_face = findViewById(R.id.imageView);
        img_face.setImageResource(R.drawable.face);

        ImageView img_face2 = findViewById(R.id.imageView2);
        img_face2.setImageResource(R.drawable.not_face);

        Button btn = findViewById(R.id.button);
        btn.setEnabled(false);

        iv_display = findViewById(R.id.iv_display);
    }

    public void showResizedPicture(View v) {
        Bitmap bmp = ((BitmapDrawable)((ImageView)v).getDrawable()).getBitmap();
        Bitmap resized = Bitmap.createScaledBitmap(bmp, 64, 64, true);
        iv_display.setImageBitmap(bmp);

        currentBitmap = bmp;

        Button btn = findViewById(R.id.button);
        btn.setEnabled(true);
    }

    public void queryServer(final View v) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        currentBitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
        IP = ((EditText)findViewById(R.id.tf_IP)).getText().toString();
        final byte[] byteArray = byteArrayOutputStream.toByteArray();

        Thread t1 = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    Socket socket = new Socket(IP, 5007);
                    DataOutputStream os = new DataOutputStream(socket.getOutputStream());
                    os.writeInt(byteArray.length);
                    os.flush();

                    BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                    in.readLine();

                    os.write(byteArray);
                    os.flush();

                    while(result == null)
                        result = in.readLine();
                    in.close();

                    os.close();
                    socket.close();
                }
                catch(Exception e) {
                    System.err.println(e);
                }
            }
        });
        t1.start();
        try {
            t1.join();
        }
        catch (InterruptedException e)
        {
            System.err.println(e);
        }
        ((TextView) findViewById(R.id.tv_bitmap)).setText(result);
        result = null;
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == 1888 && resultCode == Activity.RESULT_OK) {
            Bitmap photo = (Bitmap) data.getExtras().get("data");
            iv_display.setImageBitmap(photo);
            currentBitmap = photo;

            findViewById(R.id.button).setEnabled(true);
        }
    }
}
