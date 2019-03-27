# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask_datepicker import datepicker

from datetime import datetime
from elasticsearch import Elasticsearch

from get_mid_cpu import es_test
from flask_wtf import Form
from wtforms.fields import DateField

from jinja2 import Environment, PackageLoader, select_autoescape


app = Flask(__name__)
app.debug = True

Bootstrap(app)
datepicker(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('query_test.html')


@app.route('/comment', methods=['GET', 'POST'])
def query_test():
    es_test()
    return render_template('query_test.html')
