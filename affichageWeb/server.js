const http = require('http');
const url = require('url');
const fs = require('fs');

var server = http.createServer(function (req, res) {
    var page = url.parse(req.url).pathname;
    console.log(page);
    res.writeHead(200, { "Content-Type": "text/html" });
    if (page == '/index.js') {
        fs.readFile(__dirname + '/index.js', function (err, data) {
            if (err) {
                throw err;
            } else {
                res.write(data);
            }

        });
    }
    else {
        res.write("<p>Not found.</p>")
    }
    res.end();
});
server.listen(8080);

