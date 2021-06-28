=======
安装
=======

这篇文章介绍了如何在不同的硬件和软件环境下，安装最合适的OKNLP版本。


选择系统环境
=====================
.. raw:: html

    <style>
    .install-row {
        display: flex;
        flex-direction: row;
    }
    .install-row .install-selector-name {
        width: 7rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        color: #0F7390;
    }
    .install-row .install-selector-name span {
        border-bottom: .05rem dashed #373737;
        padding: .5rem 0;
    }
    .install-row .install-selector {
        flex-grow: 1;
        display: flex;
        flex-direction: row;
    }
    .install-row .install-selector span {
        width: 100%;
        display: inline-block;
        border: .1rem solid #1185A7;
        text-align: center;
        margin: .4rem .8rem;
        color: #1185A7;
        cursor: pointer;
        padding: .5rem 0;
    }
    .install-row .install-selector span.active {
        background-color: #1185A7;
        color: #FFFFFF;
        cursor: auto;
    }
    .install-row .install-selector span:not(.active):hover {
        text-shadow: 0 .05rem 1rem rgba(17, 133, 167, .08), .05rem 0 1rem rgba(17, 133, 167, .08);
        box-shadow: 0 .05rem 1rem rgba(17, 133, 167, .08), .05rem 0 1rem rgba(17, 133, 167, .08);
        border-color: #41b9dd;
    }
    #instructions {
        margin: 1rem 0; 
        padding: 1rem; 
        border: .1rem solid #DDD; 
        border-radius: .2rem;
    }
    #instructions pre {
        background-color: #f8f8f8; 
        font-family: monospace;
        padding: .6rem .6rem;
    }
    #instructions a {
        color: #1185A7;
        font-weight: normal;
    }
    </style>
    <div id="version-selector" style="display: flex; flex-direction: column; width: 40rem;">
        <div id="os" class="install-row">
            <div class="install-selector-name">
                <span>操作系统</span>
            </div>
            <div class="install-selector">
                <span group="os" value="windows">Windows</span>
                <span group="os" value="linux">Linux</span>
                <span group="os" value="mac">Mac OS X</span>
            </div>
        </div>
        <div id="package" class="install-row">
            <div class="install-selector-name">
                <span>安装方式</span>
            </div>
            <div class="install-selector">
                <span group="package" value="pip">Pip</span>
                <span group="package" value="conda">Conda</span>
                <span group="package" value="source">Source</span>
            </div>
        </div>
        <div id="platform" class="install-row">
            <div class="install-selector-name">
                <span>硬件平台</span>
            </div>
            <div class="install-selector">
                <span group="platform" value="cpu">CPU</span>
                <span group="platform" value="cu102">CUDA 10.2</span>
                <span group="platform" value="cu110">CUDA 11.X</span>
            </div>
        </div>
    </div>
    <div id="instructions">
        <div id="pip-cpu">
            <h3>安装指令</h3>
            <pre>pip install onnxruntime oknlp</pre>
        </div>
        
        <div id="conda-cpu">
            <h3>安装指令</h3>
            <pre>pip install onnxruntime oknlp</pre>
        </div>

        <div id="source-cpu">
            <h3>安装指令</h3>
            <pre>pip install onnxruntime
    git clone https://github.com/PLNUHT/oknlp.git
    cd oknlp
    python setup.py install</pre>
        </div>

        <div id="pip-cu102">
            <h3>安装指令</h3>
            <pre>pip install onnxruntime-gpu==1.6.0 oknlp</pre>
            <h3>环境依赖</h3>
            <ul>
                <li>CUDA: 10.2 <a href="https://developer.nvidia.com/cuda-10.2-download-archive" target="_blank">下载地址</a></li>
                <li>cuDNN: 8 <a href="https://developer.nvidia.com/rdp/cudnn-archive" target="_blank">下载地址</a></li>
            </ul>
        </div>

        <div id="pip-cu110">
            <h3>安装指令</h3>
            <pre>pip install onnxruntime-gpu==1.7.0 oknlp</pre>
            <h3>环境依赖</h3>
            <ul>
                <li>CUDA: 11.X <a href="https://developer.nvidia.com/cuda-downloads" target="_blank">下载地址</a></li>
                <li>cuDNN: 8 <a href="https://developer.nvidia.com/rdp/cudnn-archive" target="_blank">下载地址</a></li>
            </ul>
        </div>

        <div id="conda-cu102">
            <h3>安装指令</h3>
            <pre>conda install -c conda-forge cudatoolkit=10.2 cudnn=8
    pip install onnxruntime-gpu==1.6.0 oknlp</pre>
        </div>

        <div id="conda-cu110">
            <h3>安装指令</h3>
            <pre>conda install -c conda-forge cudatoolkit=11 cudnn=8
    pip install onnxruntime-gpu==1.7.0 oknlp</pre>
        </div>

        <div id="mac-cuda">
            Mac OS X系统目前不支持CUDA版本。
        </div>

        <div id="source-cu102">
            <h3>安装指令</h3>
            <pre>pip install onnxruntime-gpu==1.6.0
    git clone https://github.com/PLNUHT/oknlp.git
    cd oknlp
    python setup.py install</pre>
            <h3>环境依赖</h3>
            <ul>
                <li>CUDA: 10.2 <a href="https://developer.nvidia.com/cuda-10.2-download-archive" target="_blank">下载地址</a></li>
                <li>cuDNN: 8 <a href="https://developer.nvidia.com/rdp/cudnn-archive" target="_blank">下载地址</a></li>
            </ul>
        </div>

        <div id="source-cu110">
            <h3>安装指令</h3>
            <pre>pip install onnxruntime-gpu==1.7.0
    git clone https://github.com/PLNUHT/oknlp.git
    cd oknlp
    python setup.py install</pre>
            <h3>环境依赖</h3>
            <ul>
                <li>CUDA: 11.X <a href="https://developer.nvidia.com/cuda-downloads" target="_blank">下载地址</a></li>
                <li>cuDNN: 8 <a href="https://developer.nvidia.com/rdp/cudnn-archive" target="_blank">下载地址</a></li>
            </ul>
        </div>

    </div>
    <script>
    (function(){
        var options = {};
        function update_instructions() {
            var os = options["os"];
            var package = options["package"];
            var platform = options["platform"];
            var show_name = null;
            if (os && package && platform) {
                if (platform == "cpu") {
                    show_name = package + "-" + platform;
                } else {
                    if (os == "mac") {
                        show_name = "mac-cuda";
                    }
                    else show_name = package + "-" + platform;
                }
            }
            document.querySelectorAll("#instructions > div").forEach(function(element) {
                element.style.display = "none";
            });
            if (show_name != null) {
                document.querySelector("#instructions").style.display = "block";
                document.querySelector("#instructions > div#" + show_name).style.display = "block";
            } else {
                document.querySelector("#instructions").style.display = "none";
            }
        };
        document.querySelectorAll(".install-selector span").forEach(function(element) {
            element.addEventListener("click", function() {
                var group = element.getAttribute("group");
                var name = element.getAttribute("value");
                options[group] = name;
                document.querySelectorAll(".install-selector span[group=\"" + group + "\"]").forEach(function(element_in_group) {
                    if (element_in_group.getAttribute("value") == name) element_in_group.classList.add("active");
                    else element_in_group.classList.remove("active");
                });
                update_instructions();
            });
        });
        update_instructions();
    })();
    </script>

支持的 Python 版本
=====================

* Python 3.6
* Python 3.7
* Python 3.8
* Python 3.9