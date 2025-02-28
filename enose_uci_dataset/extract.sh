for zip in *.zip; do
  unzip -d "${zip%.zip}" "$zip"
done
