package com.example.wwydm.notfaceandroid;

import android.Manifest;
import android.app.Activity;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.provider.MediaStore;
import android.support.annotation.RequiresApi;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.transition.Explode;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.EditText;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLEncoder;
import java.text.SimpleDateFormat;
import java.util.Base64;
import java.util.Date;
import java.util.Locale;

public class faceDetectionServer extends AppCompatActivity {
    ImageView iv_display;
    TextView tv_bitmap;
    TextView tv_status;
    GridView gv_imagesView;
    GridViewAdapter a_imagesViewAdapter;
    Bitmap currentBitmap;
    String result;
    String IP;

    private Handler handler;
    private static final int ERROR = -1;
    private static final int COMPRESSING_IMAGE = 0;
    private static final int SENDING_IMAGE = 1;
    private static final int AWAITING_RESPONSE = 2;
    private static final int ANSWER_RECEIVED = 10;

    class resultFetcher implements Runnable {
        @RequiresApi(api = Build.VERSION_CODES.O)
        private String getCompressedImage() {
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            currentBitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
            byte[] bytes = byteArrayOutputStream.toByteArray();
            return Base64.getEncoder().encodeToString(bytes);
        }

        @RequiresApi(api = Build.VERSION_CODES.O)
        @Override
        public void run() {
            IP = ((EditText)findViewById(R.id.tf_IP)).getText().toString();

            handler.obtainMessage(COMPRESSING_IMAGE).sendToTarget();

            String imageB64 = getCompressedImage();
            String postData = null;
            String type = "application/x-www-form-urlencoded";
            String encodedData = null;
            OutputStream out = null;

            try {
                // Encode data"image="+imageB64;
                encodedData = URLEncoder.encode("image", "UTF-8") + "=" + URLEncoder.encode(imageB64, "UTF-8");

                // Form server connection
                URL u = new URL("http://"+IP+":5007");
                HttpURLConnection conn = (HttpURLConnection) u.openConnection();
                conn.setDoOutput(true);
                conn.setRequestMethod("POST");

                // Create output streams
                out = new BufferedOutputStream(conn.getOutputStream());
                BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out, "UTF-8"));

                // Connect to server
                conn.connect();

                // Send data
                handler.obtainMessage(SENDING_IMAGE).sendToTarget();
                writer.write(encodedData);
                writer.flush();
                writer.close();
                out.close();

                // Read response
                handler.obtainMessage(AWAITING_RESPONSE).sendToTarget();
                BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                String inputLine;
                StringBuffer response = new StringBuffer();
                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
                String final_response = response.toString().replace("\"", "");

                byte[] b = Base64.getDecoder().decode(final_response);

                currentBitmap = BitmapFactory.decodeByteArray(b, 0, b.length);
                handler.obtainMessage(ANSWER_RECEIVED).sendToTarget();
            }
            catch (MalformedURLException err) {
                Log.e("faceDetectionServer/connectionError", err.getMessage());
            }
            catch (Exception e){
                Log.e("faceDetectionServer", e.getMessage());
            }
        }

    }

    class UIHandler extends Handler {
        private int previous;
        @Override
        public void handleMessage(Message msg) {
            if (msg.what == AWAITING_RESPONSE && previous!=SENDING_IMAGE)
                return;
            previous = msg.what;

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
                case ANSWER_RECEIVED:
                    tv_status.setText(R.string.server_face);
                    iv_display.setImageBitmap(currentBitmap);
                    break;
                case ERROR:
                    tv_status.setText(R.string.server_error);
                    break;
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        getWindow().requestFeature(Window.FEATURE_CONTENT_TRANSITIONS);
        getWindow().requestFeature(Window.FEATURE_ACTIVITY_TRANSITIONS);
        super.onCreate(savedInstanceState);
        getWindow().setEnterTransition(new Explode());

        setContentView(R.layout.activity_face_detection_server);
        handler = new UIHandler();

        tv_status = findViewById(R.id.tv_status);
        gv_imagesView = findViewById(R.id.gv_Images);
        a_imagesViewAdapter = new GridViewAdapter(this);

        gv_imagesView.setAdapter(a_imagesViewAdapter);

        initializeGUI();
    }

    private void initializeGUI() {
        if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    33213);
        }

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        FloatingActionButton fab = findViewById(R.id.fab);

        fab.setBackgroundColor(getResources().getColor(R.color.colorPrimary, getTheme()));
        fab.setOnClickListener(new photoTakerListener(this));

        iv_display = findViewById(R.id.iv_display);

        gv_imagesView.setOnItemClickListener(new AdapterView.OnItemClickListener() {

            @Override
            public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
                Bitmap bmp = a_imagesViewAdapter.getItem(i);

                iv_display.setImageBitmap(bmp);
                findViewById(R.id.button).setEnabled(true);

                currentBitmap = bmp;
            }
        });
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

    private class photoTakerListener implements View.OnClickListener {
        Context uriContext;

        private photoTakerListener(Context c){
            uriContext = c;
        }

        @Override
        public void onClick(View v) {
            File photoFile = null;
            ContentValues values = new ContentValues(1);
            values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpg");
            Uri mCameraTempUri = uriContext.getContentResolver()
                    .insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

            try {
                photoFile = createImageFile();
            } catch (IOException ex) {
                Log.e("notFaceFileCreate", "Error while creating file");
            }
            if (photoFile != null) {
                Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                cameraIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                        | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);

                cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, mCameraTempUri);
                startActivityForResult(cameraIntent, 1888);
            }
        }
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == 1888 && resultCode == Activity.RESULT_OK) {
            // Refresh grid view to show newest picture
            a_imagesViewAdapter.refresh();

            // Read the image we have just taken
            String image_path = a_imagesViewAdapter.getNewestPicture();
            Bitmap image = BitmapFactory.decodeFile(image_path);

            // Rotate if needed
            try {
                image = fixImageRotation(image_path, image);
            }
            catch (IOException e) {
                Log.e("notFaceImageRotation", "Image rotation gone wrong");
            }

            //Refresh again, to show rotated picture
            gv_imagesView.invalidateViews();

            // Set displayed image to the one we have just taken
            iv_display.setImageBitmap(image);
            currentBitmap = image;

            // Finally, enable button to allow user to send picture
            findViewById(R.id.button).setEnabled(true);
        }
    }

    private Bitmap fixImageRotation(String path, Bitmap bitmap) throws IOException {
        ExifInterface ei = new ExifInterface(path);
        int orientation = ei.getAttributeInt(ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_UNDEFINED);

        Bitmap rotatedBitmap = null;
        switch(orientation) {

            case ExifInterface.ORIENTATION_ROTATE_90:
                rotatedBitmap = rotateImage(bitmap, 90);
                break;

            case ExifInterface.ORIENTATION_ROTATE_180:
                rotatedBitmap = rotateImage(bitmap, 180);
                break;

            case ExifInterface.ORIENTATION_ROTATE_270:
                rotatedBitmap = rotateImage(bitmap, 270);
                break;

            case ExifInterface.ORIENTATION_NORMAL:
            default:
                rotatedBitmap = bitmap;
        }

        // Overwrite our image
        rotatedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, new FileOutputStream(path));

        return rotatedBitmap;
    }

    private static Bitmap rotateImage(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(),
                matrix, true);
    }

    private void setDisplayedImage(String path) {
        Bitmap image = BitmapFactory.decodeFile(path);
        iv_display.setImageBitmap(image);
    }

    private File createImageFile() throws IOException{
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.ENGLISH).format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image;

        image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",   /* suffix */
                storageDir      /* directory */
        );

        return image;
    }
}
