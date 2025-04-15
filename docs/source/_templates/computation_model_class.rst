{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block methods %}
    {% if methods %}

    .. rubric:: Methods

    .. autosummary::
        :toctree: classmethods
        :template: method.rst

        {{ objname }}.analyze_physics
        {{ objname }}.fit
        {{ objname }}.predict

    {% endif %}
    {% endblock %}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: Properties

    .. autosummary::
    {% for item in attributes %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}
