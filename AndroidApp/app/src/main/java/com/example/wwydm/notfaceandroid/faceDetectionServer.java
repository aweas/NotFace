package com.example.wwydm.notfaceandroid;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.design.widget.FloatingActionButton;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.Socket;

public class faceDetectionServer extends AppCompatActivity {
    ImageView iv_display;
    TextView tv_bitmap;
    TextView tv_status;
    Bitmap currentBitmap;
    String result;
    String IP;

    private Handler handler;
    private static final int ERROR = -1;
    private static final int COMPRESSING_IMAGE = 0;
    private static final int SENDING_IMAGE = 1;
    private static final int AWAITING_RESPONSE = 2;
    private static final int ANSWER_FACE = 10;
    private static final int ANSWER_NOT_FACE = 11;

    class resultFetcher implements Runnable {
        @Override
        public void run() {
            IP = ((EditText)findViewById(R.id.tf_IP)).getText().toString();

            handler.obtainMessage(COMPRESSING_IMAGE).sendToTarget();

            final byte[] byteArray = getCompressedImage();

            try {
                handler.obtainMessage(SENDING_IMAGE).sendToTarget();

                Socket socket = new Socket(IP, 5007);
                DataOutputStream os = new DataOutputStream(socket.getOutputStream());
                os.writeInt(byteArray.length);
                os.flush();

                BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                in.readLine();

                os.write(byteArray);
                os.flush();

                handler.obtainMessage(AWAITING_RESPONSE).sendToTarget();
                while(result == null)
                    result = in.readLine();

                if(result.equals("Face"))
                    handler.obtainMessage(ANSWER_FACE).sendToTarget();
                else if(result.equals("notFace"))
                    handler.obtainMessage(ANSWER_NOT_FACE).sendToTarget();

                result = null;
                in.close();

                os.close();
                socket.close();
            }
            catch(Exception e) {
                System.err.println(e);
                handler.obtainMessage(ERROR).sendToTarget();
            }
        }
    }

    private byte[] getCompressedImage() {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        currentBitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
        return byteArrayOutputStream.toByteArray();
    }

    class UIHandler extends Handler {
        @Override
        public void handleMessage(Message msg) {
            switch (msg.what)
            {
                case COMPRESSING_IMAGE:
                    tv_status.setText(R.string.server_compress);
                    break;
                case SENDING_IMAGE:
                    tv_status.setText(R.string.server_send);
                    break;
                case AWAITING_RESPONSE:
                    tv_status.setText(R.string.server_waiting);
                    break;
                case ANSWER_FACE:
                    tv_status.setText(R.string.server_face);
                    tv_bitmap.setText(R.string.server_ans_face);
                    break;
                case ANSWER_NOT_FACE:
                    tv_status.setText(R.string.server_notFace);
                    tv_bitmap.setText(R.string.server_ans_notFace);
                    break;
                case ERROR:
                    tv_status.setText(R.string.server_error);
                    break;
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_face_detection_server);
        handler = new UIHandler();
        initializeGUI();
    }

    private void initializeGUI() {
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        FloatingActionButton fab = findViewById(R.id.fab);
        fab.setBackgroundColor(getResources().getColor(R.color.colorPrimary));
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

        iv_display = findViewById(R.id.iv_display);
        tv_bitmap = findViewById(R.id.tv_bitmap);
        tv_status = findViewById(R.id.tv_status);
    }

    public void showPicture(View v) {
        Bitmap bmp = ((BitmapDrawable)((ImageView)v).getDrawable()).getBitmap();
        iv_display.setImageBitmap(bmp);

        currentBitmap = bmp;

        Button btn = findViewById(R.id.button);
        btn.setEnabled(true);
    }

    public void queryServer(final View v) {
        Thread t1 = new Thread(new resultFetcher());
        t1.start();
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
