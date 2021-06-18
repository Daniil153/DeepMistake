pip install git+https://github.com/Koziev/rutokenizer > /dev/null \
  && pip install git+https://github.com/Koziev/rupostagger > /dev/null \
  && pip install -r requirements.txt > /dev/null \
  && git clone https://github.com/Koziev/ruword2tags.git > /dev/null \
  && rm -rf /ruword2tags/ruword2tags/ruword2tags.db \
  && gdown --id "1xlL8ijnwE6tAPpsil7Q1yWkXY4mn2YCd" \
  && mv ruword2tags.db ruword2tags/ruword2tags/ \
  && cd ruword2tags \
  && python setup.py install > /dev/null \
  && pip install git+https://github.com/Koziev/rulemma > /dev/null \
  && cd ../