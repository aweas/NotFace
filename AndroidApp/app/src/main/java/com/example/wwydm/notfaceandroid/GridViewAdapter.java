package com.example.wwydm.notfaceandroid;

import android.app.Activity;
import android.content.Context;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.provider.MediaStore;
import android.text.Layout;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.ImageView;

import java.util.ArrayList;

public class GridViewAdapter extends BaseAdapter {
    private Context context;
    private ArrayList<String> data;
    private Bitmap img;
    private Layout layoutResourceId;
    int lastSelected = 0;

    GridViewAdapter(Context context) {
        this.context = context;

        data = getAllShownImagesPath();

        BitmapFactory.Options options = new BitmapFactory.Options();
        img = BitmapFactory.decodeFile(data.get(0));
    }

    public void refresh() {
        data = getAllShownImagesPath();
    }

    public String getNewestPicture() {
        return data.get(0);
    }

    @Override
    public long getItemId(int position) {
        return 0;
    }

    @Override
    public Bitmap getItem(int position) {
        return BitmapFactory.decodeFile(data.get(position));
    }

    @Override
    public int getCount() {
        return data.size();
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        // Inflate view if needed
        if (convertView == null) {
            final LayoutInflater inflater = ((Activity) context).getLayoutInflater();
            convertView = inflater.inflate(R.layout.image_layout, parent, false);
        }

        // Get convertView's children
        final ImageView picture = (ImageView)convertView.findViewById(R.id.img);

        // Fill children
        picture.setImageBitmap(getBitmap(position));
//        picture.setImageBitmap(img);
        return convertView;
    }

    private Bitmap getBitmap(int position){
        // Read bitmap from path
        BitmapFactory.Options options = new BitmapFactory.Options();
        Bitmap bitmap = BitmapFactory.decodeFile(data.get(position));

        // Fit bitmap to square
        int height, width;
        int x_start = 0, y_start = 0;

        if(bitmap.getHeight() > bitmap.getWidth()) {
            height = width = bitmap.getWidth();
            y_start = (bitmap.getHeight()-height)/2;
        }
        else{
            height = width = bitmap.getHeight();
            x_start = (bitmap.getWidth()-width)/2;
        }

        bitmap = Bitmap.createBitmap(bitmap, x_start, y_start, width, height);

        return bitmap;
    }

    private ArrayList<String> getAllShownImagesPath() {
        Uri uri;
        Cursor cursor;
        int column_index_data, column_index_folder_name;
        ArrayList<String> listOfAllImages = new ArrayList<>();
        String absolutePathOfImage;

        uri = android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI;

        String[] projection = { MediaStore.MediaColumns.DATA,
                MediaStore.Images.Media.BUCKET_DISPLAY_NAME };

        cursor = this.context.getContentResolver().query(uri, projection, null,
                null, MediaStore.MediaColumns.DATE_ADDED + " DESC");

        column_index_data = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.DATA);
        column_index_folder_name = cursor
                .getColumnIndexOrThrow(MediaStore.Images.Media.BUCKET_DISPLAY_NAME);

        while (cursor.moveToNext()) {
            absolutePathOfImage = cursor.getString(column_index_data);

            listOfAllImages.add(absolutePathOfImage);
        }
        return listOfAllImages;
    }

    public class ViewHolder extends View{
        private ImageView img;
        public ViewHolder(Context context) {
            super(context);

            img = new ImageView(context);
        }
    }
}
