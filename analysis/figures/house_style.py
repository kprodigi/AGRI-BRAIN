"""Shared look of the figures: Arial Bold throughout, a bold figure title, the legend above the panels, black-edged
hatched bars, light horizontal grid, left and bottom axes only.

Colours and hatches carry one meaning across all figures:
policies   Static #595959 (plain), No-Context #0072B2 ('..'), AGRI-BRAIN #009E73 ('xx')
routes     cold chain #332288 (plain), local redistribution #009E73 ('..'), recovery #D55E00 ('xx')
context    tool (MCP) #D55E00 ('//'), retrieval #882255 ('..')
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

STATIC, NOCTX, AGB = '#595959', '#0072B2', '#009E73'
POLICY = {'static': (STATIC, ''), 'no_context': (NOCTX, '..'), 'agribrain': (AGB, 'xx')}
ROUTE = {'Cold chain': ('#332288', ''), 'Local redistribution': ('#009E73', '..'), 'Recovery': ('#D55E00', 'xx')}
TOOL, RETRIEVAL, RULE = '#D55E00', '#882255', '#D55E00'
WIDTH = 13.0          # inches; saved at 392 dpi, the submitted figures are 5100 px wide
DPI = 392

plt.rcParams.update({
    'font.family': 'Arial', 'font.weight': 'bold', 'font.size': 12,
    'axes.labelsize': 13, 'axes.labelweight': 'bold', 'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'figure.titlesize': 18, 'figure.titleweight': 'bold', 'legend.fontsize': 12,
    'axes.edgecolor': '0.35', 'axes.linewidth': 1.2, 'xtick.color': 'black', 'ytick.color': 'black',
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2, 'xtick.major.size': 5, 'ytick.major.size': 5,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'hatch.linewidth': 1.0,
    'mathtext.default': 'bf', 'axes.titlepad': 8, 'axes.titlelocation': 'center',
})


def axes_style(ax, grid='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color='0.9', linewidth=1.0)
    ax.set_axisbelow(True)


def patch(color, hatch, label):
    return Patch(facecolor=color, hatch=hatch, edgecolor='black', linewidth=0.8, label=label)


def bar_kw(color, hatch):
    return dict(color=color, hatch=hatch, edgecolor='black', linewidth=0.8)


def title_and_legend(fig, title, handles, ncol, legend_y=0.935, title_y=1.0):
    fig.suptitle(title, y=title_y)
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, legend_y), ncol=ncol, frameon=False,
               handlelength=2.2, handleheight=1.1, columnspacing=2.2)


CANVAS = 13.3        # inches: every PNG is padded to this width, so a point prints at one size in every figure


def save(fig, stem):
    fig.savefig(f'{stem}.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    from PIL import Image
    im = Image.open(f'{stem}.png').convert('RGB')
    # without Arial a wider substitute font can exceed the canvas; the figure is then kept at its own width
    target = max(round(CANVAS * DPI), im.width)
    canvas = Image.new('RGB', (target, im.height), 'white')
    canvas.paste(im, ((target - im.width) // 2, 0))
    canvas.save(f'{stem}.png', dpi=(DPI, DPI))
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('written', stem)
