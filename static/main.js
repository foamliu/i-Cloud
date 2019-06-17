$(document).ready( function() {
  	this.canvas = document.getElementById('canvas');
  	this.cube = null;

  	initWebGL();
	});

function initWebGL() {
    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera( 45, this.canvas.width / this.canvas.height, 0.1, 1000 );

    var canvas = document.getElementById('canvas');
    var renderer = new THREE.WebGLRenderer({canvas: canvas});
    renderer.setSize( canvas.width, canvas.height );

    var geometry = new THREE.BoxGeometry( 1, 1, 1 );
    var material = new THREE.MeshNormalMaterial();
    var cube = new THREE.Mesh( geometry, material );
    scene.add( cube );
    this.cube = cube;

    camera.position.z = 5;

    var render = function () {
        requestAnimationFrame( render );

        //cube.rotation.x += 0.01;
        //cube.rotation.y += 0.01;
        //cube.rotation.z += 0.01;

  	    this.cube.rotation.x = this.pitch;
		this.cube.rotation.y = -this.roll;
		this.cube.rotation.z = this.yaw;

        renderer.render(scene, camera);
    };

    render();
}
