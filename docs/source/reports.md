## Creating HTML reports
It is possible to create an HTML report to display images, tables and text, for example to display the results of evaluating your model. To do this you will need to create two files:
1. A report config file
2. A script for report creation

### 1. Creating a report config
Your report config should live in a yaml file. It should contain the key "report_contents" followed by a list of components to be added to the report.
These can be one of

- text
- image
- image gallery (multiple images on one plot)
- table (loaded from a csv file)

 The components are defined as dictionaries which, as a minimum, have the keys
 - "type" - where the value represents the data type {"image"/"image_gallery"/"table"/"text"} as described above
- "value" - where the value will be a path to an image file, folder or csv file, or a string in the case of a text component

For example, your config may look something like this:

```yaml
report_contents:
    - {"type": "text", "value": "Subheading 1"}
    - {"type": "image", "value": "<path_to_image>.png"}
    - {"type": "image_gallery", "value": "<path_to_image_folder>"}
    - {"type": "image_gallery", "value":"<path_to_img1>.jpg, <path_to_img3>.jpg, <path_to_img4>.jpg"}
    - {"type": "table", "value": "<path_to_data>.csv"}
```
You can have as many of these componets as you wish


### Breakdown of text component


```yaml
- {"type": "text", "value": "Subheading 1"}

```
To include a text component on your report, the entry "type" must equal "text", and the entry "value" must be a string containing the text you want to display. You can include HTML tags here, for example `<h1>` to create a title/subtitle.

### Breakdown of image component

```yaml
- {"type": "image", "value": "<path_to_image>.png", ["figsize" : [<int>, <int>]]}
```
To include an image component on your report, the entry "type" must equal "image", and the entry "value" must be a string representing the path to the image file. The format can be any image format supported by your browser. The entry ["figsize"](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html) is optional, and can be used to specify the size of the plot on your report, in inches.

Note that including consecutive image components in your report_contents will result in multiple images, each starting on a new line. If you wish to arrange these images side by side, consider using an image_gallery component, with two or more columns.


### Breadown of image_gallery component

```yaml
- {"type": "image_gallery", "value": "<path_to_img_folder>", ["num_cols": 4], ["figsize": [20, 20]]}
```
To include an image gallery component on your report (i.e. multiple images arranged as subplots on the same plot), the entry "type" must equal "image_gallery", and the entry "value" must be a string representing either
- a path to a folder containing multiple image files (these will all be plotted)
- a string containing multiple image file paths, separated by commas

The format of these images can be any image format supported by [PIL.open](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open). The entry ["figsize"](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html) is optional, and can be used to specify the size of the overall lot on your report, in inches.

The entry "num_cols" is also optional, and can be used to determine how many colums to add to your plot - i.e., how many subplots to plot side by side. The number of rows is determined based on the overall number of images, and the number of columns specified (by default this is 2)


### Breakdown of table type

TBC

### To instantiate a new report:

TBC