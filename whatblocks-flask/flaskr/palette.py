import os
from werkzeug.utils import secure_filename
import functools
from . import segmentation
import sys

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, current_app
)
from flaskr.db import get_db

bp = Blueprint('palette', __name__, url_prefix='/palette')

def upload_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('auth.login'))

        return view(**kwargs)

    return wrapped_view


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_filetype(filename):
    return '.' in filename and \
           filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS



@bp.route('/upload', methods=['GET', 'POST'])
def upload():
    print('entering upload bp', file=sys.stdout) 
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_filetype(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('palette.extract', name=filename))
        
    return render_template('palette/upload.html')

@bp.route('/result', methods=['GET', 'POST'])
def result():
    print('extraction result view')
    return render_template('palette/result.html')

@bp.route('/extract/<name>')
def extract(name):
    print('extractin', file=sys.stdout)   
    extracted, seg_filename = segmentation.extractPalette(current_app.config['UPLOAD_FOLDER'], name, current_app.config['SEG_FOLDER'])
    
    return render_template('palette/result.html', extracted=extracted, uploaded_name=name, seg_filename=seg_filename)



@bp.route('/texture/<name>')
def get_mc_texture(name):
    return redirect(url_for('static', filename = 'mc-textures/' + name + '.png' ), code=301)

@bp.route('/uploaded/<filename>')
def get_uploaded(filename):
    print("uploaded filename: " + filename, file=sys.stdout)   
    return redirect(url_for('static', filename = 'uploads/' + filename ), code=301)

@bp.route('/seg/<filename>')
def get_seg(filename):
    print("seg filename: " + filename, file=sys.stdout)   
    return redirect(url_for('static', filename = 'seg/' + filename ), code=301)
