function load(url) {
  var head = document.getElementsByTagName('head');
  var js = document.createElement('script');
  js.src = url;
  head[0].appendChild(js);
}

function get_json(url) {
  var xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function() {
    df = JSON.parse(xhr.responseText);
    console.log(df.length);
  };
  xhr.open("GET", url, true);
  xhr.send(null);
}
