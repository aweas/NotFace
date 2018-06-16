package com.example.wwydm.notfaceandroid;

import android.app.Activity;
import android.content.AsyncTaskLoader;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.support.annotation.NonNull;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

/**
 * Created by wwydm on 24.05.2018.
 * Populates ImageOverview view
 */

public class ImageOverviewAdapter extends RecyclerView.Adapter<ImageOverviewAdapter.cardedImageHolder> {
    private Context context;
    private ArrayList<Bitmap> image_to_set;

    public class cardedImageHolder extends RecyclerView.ViewHolder {

        private ImageView img;
        public cardedImageHolder(View v) {
            super(v);

            img = (ImageView)(v.findViewById(R.id.img));
        }
    }

    ImageOverviewAdapter(Context context) {
        image_to_set = new ArrayList();

        this.context = context;
//        this.image_to_set = BitmapFactory.decodeFile("/storage/emulated/0/DCIM/Facebook/FB_IMG_1527169943950.jpg");
        String[] URLs = new String[3];
        URLs[0] = "https://uploads3.wikiart.org/images/leon-arthur-tutundjian/la-boule-noire-1926.jpg";
        URLs[1] = "https://uploads1.wikiart.org/images/arshile-gorky/the-raven-composition-no-3.jpg";
        URLs[2] = "https://uploads6.wikiart.org/images/arshile-gorky/blue-figure-in-a-chair.jpg";

        for(String u: URLs) {
            new BitmapDownloader().execute(u);
        }
    }

    @Override
    public cardedImageHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        LayoutInflater inflater = ((Activity) context).getLayoutInflater();
        View itemView = inflater.inflate(R.layout.image_card_mini, parent, false);

        return new cardedImageHolder(itemView);
    }

    @Override
    public void onBindViewHolder(cardedImageHolder holder, int position) {
//            double scale = holder.img.getMeasuredWidth()/image_to_set.getWidth();
//            int width = (int) (image_to_set.getWidth() * scale);
//            int height = (int) (image_to_set.getHeight() * scale);
//
//            image_to_set = Bitmap.createScaledBitmap(image_to_set, width, height, false);
        holder.img.setImageBitmap(image_to_set.get(position));

    }

    @Override
    public int getItemCount() {
        return image_to_set.size();
    }

    private class BitmapDownloader extends AsyncTask {
        @Override
        protected Bitmap doInBackground(Object[] objects) {
            try {
                java.net.URL url = new java.net.URL((String)objects[0]);
                HttpURLConnection connection = (HttpURLConnection) url
                        .openConnection();
                connection.setDoInput(true);
                connection.connect();

                InputStream input = connection.getInputStream();

                return BitmapFactory.decodeStream(input);
            } catch (IOException e) {
                e.printStackTrace();
                return null;
            }
        }

        @Override
        protected void onPostExecute(Object result) {
            image_to_set.add((Bitmap)result);
            notifyDataSetChanged();
        }
    }

}
