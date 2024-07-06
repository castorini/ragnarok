tooltip_style = """
<style>
.tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
    text-decoration: underline;
}

.tooltip-body {
    visibility: hidden;
    position: absolute;
    z-index: 50;
    top: 100%;
    transform: translateX(-50%);
    width: 600px;
    color: #fff;
    background-color: rgba(0, 0, 0, 0.9);
    padding: 5px;
    border-radius: 5px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    white-space: normal;
    word-wrap: break-word;
    overflow-wrap: break-word;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
}

.tooltip:hover .tooltip-body {
  visibility: visible;
}

</style>
"""

html_content = """
<div class='navbar'>
    <h1>Ragnarök Chatbot Arena</h1>
    <p>Ask any question to RAG pipelines! Heavily built on the code for <a href="https://chat.lmsys.org">https://chat.lmsys.org</a> :)</p>
</div>
"""
