from flask import Flask, request, render_template, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import os
import cv2
from utils import read_video, save_video, preview_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/input_video'
app.config['OUTPUT_FOLDER'] = 'static/output_videos'

# Ensure the output directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Save the uploaded video
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            # Process the video
            output_filename = process_video(input_path)
            print(output_filename)
            return redirect(url_for('preview', filename=output_filename))
    return render_template('index.html')

def convert_video_to_mp4(input_path, output_path):
    command = [
        'ffmpeg', '-y', '-i', input_path,
        '-vcodec', 'libx264', '-acodec', 'aac', '-strict', 'experimental',
        output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def process_video(input_path):
    # Read Video
    video_frames = read_video(input_path)
    tracker = Tracker("models/best3.pt")

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Player assigner
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    temp_output_path = os.path.join(app.config['OUTPUT_FOLDER'], "temp_output.avi")
    save_video(output_video_frames, temp_output_path)

    # Convert to mp4
    output_filename = "processed_" + os.path.splitext(os.path.basename(input_path))[0] + ".mp4"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    convert_video_to_mp4(temp_output_path, output_path)

    # Remove the temporary file if needed
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    return output_filename

@app.route('/preview/<filename>')
def preview(filename):
    return render_template('preview.html', video_file=filename)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
