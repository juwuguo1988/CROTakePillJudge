
package org.tensorflow.lite.examples.detection.activity;

import android.content.DialogInterface;
import android.os.Bundle;


import com.baidu.idl.face.platform.FaceStatusEnum;
import com.baidu.idl.face.platform.ui.FaceLivenessActivity;

import org.tensorflow.lite.examples.detection.dialog.DefaultDialog;

import java.sql.Time;
import java.util.HashMap;
import java.util.Timer;
import java.util.TimerTask;


public class FaceLivenessExpActivity extends FaceLivenessActivity {

    private DefaultDialog mDefaultDialog;
    private Timer mTimer;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public void onLivenessCompletion(FaceStatusEnum status, String message, HashMap<String, String> base64ImageMap) {
        super.onLivenessCompletion(status, message, base64ImageMap);
        if (status == FaceStatusEnum.OK && mIsCompletion) {
            initTimeOutControl();
        } else if (status == FaceStatusEnum.Error_DetectTimeout ||
                status == FaceStatusEnum.Error_LivenessTimeout ||
                status == FaceStatusEnum.Error_Timeout) {
            showMessageDialog("活体检测", "采集超时");
        }
    }

    private void showMessageDialog(String title, String message) {
        if (mDefaultDialog == null) {
            DefaultDialog.Builder builder = new DefaultDialog.Builder(this);
            builder.setTitle(title)
                    .setMessage(message)
                    .setNegativeButton("确认",
                            new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    mDefaultDialog.dismiss();
                                    finish();
                                }
                            });
            mDefaultDialog = builder.create();
            mDefaultDialog.setCancelable(false);
        }
        mDefaultDialog.dismiss();
        mDefaultDialog.show();
    }

    @Override
    public void finish() {
        super.finish();
    }


    /**
     * 开启一个timer来维持超时控制
     */
    private void initTimeOutControl() {
        mTimer = new Timer();// 构造函数new Timer(true)
        // 表明这个timer以daemon方式运行（优先级低，程序结束timer也自动结束）。
        TimerTask timerTask = new TimerTask() {
            @Override
            public void run() {
                showMessageDialog("活体检测", "检测成功");
            }
        };
        mTimer.schedule(timerTask, 2000);
    }

}
